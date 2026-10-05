import { Component, OnDestroy, OnInit, inject } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { HttpErrorResponse } from '@angular/common/http';
import { Subscription, catchError, firstValueFrom, map, throwError, timeout } from 'rxjs';
import { ApiService, QueryEvent, QueryResponse, WorkflowEvent } from './services/api.service';
import { environment } from '../environments/environment';

const agentNames = ['Precheck', 'SchemaRetriever', 'QueryContextAgent', 'QueryGeneratorAgent', 'ValidatorAgent', 'QueryCostGuard', 'OptimizationAdvisor', 'ExecutorAgent', 'ResultsSummaryAgent'];

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent implements OnInit, OnDestroy {
  private readonly api = inject(ApiService);
  private readonly subscriptions = new Subscription();
  private healthTimer?: ReturnType<typeof setTimeout>;
  private destroyed = false;
  private detailTrigger: HTMLElement | null = null;
  private previousBodyOverflow = "";
  optimizationRequired = false;
  costAnalysis: QueryResponse['cost_analysis'];
  optimizationSuggestions: NonNullable<QueryResponse['suggestions']> = [];
  optimizationSubmitting = false;
  pendingRequestId = '';
  refinement = '';
  refining = false;
  pagination: QueryResponse['pagination'];
  optimizationMessage = '';
  question = '';
  originalQuestion = '';
  requestId = '';
  hasSubmitted = false;
  workflowExpanded = true;
  selectedRecord: unknown = null;
  detailTitle = '';
  detailFields: { label: string; value: string }[] = [];
  loading = false;
  regenerating = false;
  feedbackSubmitting = false;
  feedbackChoice: 'yes' | 'no' | null = null;
  feedbackComment = '';
  feedbackSaved = false;
  feedbackMessage = '';
  schemaReady = false;
  healthStatus = 'Checking database readiness…';
  error = '';
  runStatus = 'Ready for your question';
  executionTrace: WorkflowEvent[] = this.newTrace();
  transitions: string[] = [];
  result: QueryResponse | null = null;
  results: Record<string, unknown>[] = [];
  columns: string[] = [];
  scalarResult: string | null = null;
  summary = '';
  queryExpanded = false;
  queryText = '';
  queryOperation = '';
  queryCollection = '';
  readonly icons = { pending: '○', running: '→', completed: '✓', failed: '×', skipped: '–' };

  get busy(): boolean { return this.loading || this.regenerating || this.feedbackSubmitting || this.optimizationSubmitting; }

  ngOnInit(): void { this.checkHealth(); }
  ngOnDestroy(): void {
    this.destroyed = true;
    this.closeDetails();
    clearTimeout(this.healthTimer);
    this.subscriptions.unsubscribe();
  }
  private newTrace(): WorkflowEvent[] { return agentNames.map(agent => ({ agent, status: 'pending' })); }

  private checkHealth(): void {
    this.subscriptions.add(this.api.health().subscribe({
      next: health => {
        this.schemaReady = health.schema_ready && health.database === 'up';
        this.healthStatus = health.database !== 'up' ? 'Database unavailable' : health.schema_ready ? 'Ready' : 'Preparing database schema...';
        this.scheduleHealth();
      },
      error: () => {
        this.schemaReady = false;
        this.healthStatus = 'Backend unavailable';
        this.scheduleHealth();
      }
    }));
  }
  private scheduleHealth(): void {
    if (!this.destroyed) this.healthTimer = setTimeout(() => this.checkHealth(), this.schemaReady ? 30000 : 5000);
  }

  ask(): void {
    if (this.busy || !this.schemaReady || !this.question.trim()) return;
    this.originalQuestion = this.question.trim();
    this.runQuery(false);
  }
  shortcut(event: KeyboardEvent): void {
    if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) { event.preventDefault(); this.ask(); }
  }

  private updateWorkflow(event: WorkflowEvent): void {
    if (!agentNames.includes(event.agent)) return;
    this.executionTrace = this.executionTrace.map(step => step.agent === event.agent ? event : step);
    this.transitions.push(`${event.agent} → ${event.status}${event.elapsed_ms !== undefined ? ` (${event.elapsed_ms} ms)` : ''}`);
    if (event.status === 'running') this.runStatus = `${event.agent} is running…`;
  }

  private runQuery(regenerating: boolean): void {
    this.hasSubmitted = true; this.workflowExpanded = true; this.closeDetails();
    this.loading = true; this.regenerating = regenerating;
    this.error = ''; this.runStatus = regenerating ? 'Regenerating answer…' : 'Starting workflow…';
    this.executionTrace = this.newTrace(); this.transitions = [];
    // Keep an earlier answer visible if regeneration fails; its feedback stays associated.
    const priorResult = this.result;
    if (!regenerating) this.clearAnswer();
    const plainQuery = () => this.api.query(this.originalQuestion).pipe(map(data => ({ type: 'result', data } as QueryEvent)));
    let live = environment.useStreaming;
    const query = live ? this.api.streamQuery(this.originalQuestion).pipe(catchError(error => {
      if (error instanceof HttpErrorResponse && [404, 405].includes(error.status)) {
        live = false; this.runStatus = 'Running query; trace available on completion…'; return plainQuery();
      }
      return throwError(() => error);
    })) : plainQuery();
    this.subscriptions.add(query.pipe(timeout({ each: 300000 })).subscribe({
      next: event => {
        if (event.type === 'workflow') this.updateWorkflow(event.data);
        else { this.displayResult(event.data); if (!live) this.returnedTrace(event.data); }
      },
      error: error => {
        this.error = `${regenerating ? 'Regeneration failed. ' : ''}${this.message(error)}`;
        this.runStatus = 'Workflow interrupted'; this.finishTrace();
        this.loading = this.regenerating = false;
        if (regenerating && priorResult) this.feedbackMessage = 'Your feedback was saved. You can retry regeneration below.';
      },
      complete: () => {
        this.runStatus = this.optimizationRequired ? 'Optimization required' : this.result?.execution_success ? 'Workflow complete' : 'Workflow failed';
        if (!this.optimizationRequired) this.finishTrace(); this.workflowExpanded = this.optimizationRequired; this.loading = this.regenerating = false;
      }
    }));
  }

  private clearAnswer(): void {
    this.optimizationRequired = false; this.refining = false; this.refinement = ''; this.pagination = undefined;
    this.result = null; this.requestId = ''; this.results = []; this.columns = []; this.summary = ''; this.scalarResult = null;
    this.queryExpanded = false; this.queryText = ''; this.queryOperation = ''; this.queryCollection = '';
    this.feedbackChoice = null; this.feedbackComment = ''; this.feedbackSaved = false; this.feedbackMessage = '';
  }
  private displayResult(result: QueryResponse): void {
    this.clearAnswer(); this.result = result; this.requestId = result.request_id;
    this.pagination = result.pagination;
    if (result.status === 'optimization_required') {
      this.optimizationRequired = true; this.pendingRequestId = result.request_id;
      this.costAnalysis = result.cost_analysis; this.optimizationSuggestions = result.suggestions || [];
      this.optimizationMessage = result.message || 'This request may be expensive.';
      this.runStatus = 'Optimization required'; this.workflowExpanded = true;
      this.updateWorkflow({ agent: 'QueryCostGuard', status: 'completed', message: 'Heavy query detected' });
      this.updateWorkflow({ agent: 'OptimizationAdvisor', status: 'completed', message: 'Choose an optimization' });
      return;
    }
    const query = result.validated_query || result.generated_query;
    if (query) {
      this.queryText = JSON.stringify(query, null, 2);
      this.queryOperation = typeof query['operation'] === 'string' ? query['operation'] : '';
      this.queryCollection = typeof query['collection'] === 'string' ? query['collection'] : '';
    }
    const raw: unknown = result.execution_result;
    if (Array.isArray(raw)) {
      this.results = raw.map(value => value !== null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : { value });
      this.columns = [...new Set(this.results.flatMap(row => Object.keys(row)))];
    } else if (raw !== undefined && raw !== null) this.scalarResult = this.format(raw);
    this.summary = result.results_summary_agent_output?.summary_text || result.final_response || 'No summary available.';
    if (!result.execution_success) this.error = result.execution_error_code === 'query_timeout'
      ? 'This query exceeded the execution time limit. Narrow the filters or date range, or simplify the joins. Limiting returned rows alone may not reduce aggregation work.'
      : 'The backend could not complete this query. Please revise the question or try again.';
  }
  private finishTrace(): void {
    this.executionTrace = this.executionTrace.map(step => step.status === 'running' ? { ...step, status: 'failed' } :
      step.status === 'pending' ? { ...step, status: 'skipped', message: 'Not reached' } : step);
  }
  private returnedTrace(result: QueryResponse): void {
    const called = result.execution_path?.agents_called || [];
    const completed = new Set<string>();
    if (result.schema_spec || result.query_context) { completed.add('Precheck'); completed.add('SchemaRetriever'); }
    if (result.query_context) completed.add('QueryContextAgent');
    if (result.generated_query) completed.add('QueryGeneratorAgent');
    if (result.validated_query) completed.add('ValidatorAgent');
    if (result.execution_success) completed.add('ExecutorAgent');
    if (called.includes('summary') && result.results_summary_agent_output) completed.add('ResultsSummaryAgent');
    if (result.cost_analysis) completed.add('QueryCostGuard');
    if (result.status === 'optimization_required') {
      this.executionTrace = this.newTrace().map(step => ({ ...step, status: ['QueryCostGuard', 'OptimizationAdvisor'].includes(step.agent) ? 'completed' : ['ExecutorAgent', 'ResultsSummaryAgent'].includes(step.agent) ? 'pending' : 'completed' }));
      return;
    }
    this.executionTrace = this.newTrace().map(step => ({ ...step,
      status: completed.has(step.agent) ? 'completed' : step.agent === 'ExecutorAgent' && result.execution_error && result.validated_query ? 'failed' : 'skipped',
      message: step.agent === 'ResultsSummaryAgent' && result.execution_path?.summary_skipped ? 'Deterministic summary used' : 'Returned execution metadata'
    }));
    const names: Record<string, string> = { query_context: 'QueryContextAgent', query_generator: 'QueryGeneratorAgent', repair: 'QueryGeneratorAgent (repair)', summary: 'ResultsSummaryAgent' };
    this.transitions = called.map((agent, index) => `${index + 1}. ${names[agent] || agent}`);
  }
  async optimize(optimization: string, page = 1): Promise<void> {
    if (this.busy) return;
    if (['refine', 'narrow_date'].includes(optimization) && !this.refining) { this.refining = true; return; }
    if (['refine', 'narrow_date'].includes(optimization) && !this.refinement.trim()) return;
    this.optimizationSubmitting = true; this.error = ''; this.runStatus = 'Applying optimization...';
    const requestId = this.pendingRequestId || this.requestId;
    const pageSize = this.pagination?.page_size;
    this.transitions.push(`Optimization selected: ${optimization}`);
    this.updateWorkflow({ agent: ['refine', 'narrow_date', 'summary'].includes(optimization) ? 'QueryGeneratorAgent' : 'ValidatorAgent', status: 'running', message: 'Reusing saved context' });
    try {
      const result = await firstValueFrom(this.api.optimizeQuery(requestId, {
        optimization, page, value: pageSize, refinement: this.refinement.trim() || undefined
      }));
      if (result.request_id !== requestId && result.question) this.originalQuestion = result.question;
      this.displayResult(result);
      for (const event of result.execution_trace || []) this.updateWorkflow(event);
      if (!this.optimizationRequired) {
        this.updateWorkflow({ agent: 'ExecutorAgent', status: result.execution_success ? 'completed' : 'failed' });
        this.finishTrace(); this.runStatus = result.execution_success ? 'Workflow complete' : 'Workflow failed';
      }
    } catch (error) {
      this.error = this.message(error); this.runStatus = 'Optimization could not be completed';
      this.updateWorkflow({ agent: 'ExecutorAgent', status: 'pending' });
    } finally { this.optimizationSubmitting = false; }
  }

  label(key: string): string {
    if (key === '_id') return 'ID';
    const text = key.replace(/([a-z0-9])([A-Z])/g, '$1 $2').replace(/[_.-]+/g, ' ').trim();
    return text.charAt(0).toUpperCase() + text.slice(1);
  }
  private readableValue(value: unknown): unknown {
    if (typeof value === 'string' && /^[{[]/.test(value.trim())) {
      try { const parsed = JSON.parse(value); if (parsed !== null && typeof parsed === 'object') return parsed; } catch { /* Ordinary text stays unchanged. */ }
    }
    return value;
  }
  isStructured(value: unknown): boolean {
    value = this.readableValue(value);
    return value !== null && typeof value === 'object' && !this.isMongoScalar(value);
  }
  private isMongoScalar(value: object): boolean {
    const keys = Object.keys(value);
    return keys.length === 1 && ['$oid', '$date', '$numberDecimal', '$numberLong', '$numberInt', '$numberDouble'].includes(keys[0]);
  }
  format(value: unknown): string {
    value = this.readableValue(value);
    if (value === undefined) return '\u2014';
    if (value === null) return 'null';
    if (typeof value === 'boolean') return value ? 'Yes' : 'No';
    if (typeof value !== 'object') return String(value);
    if (this.isMongoScalar(value)) return this.format(Object.values(value)[0]);
    if (Array.isArray(value)) {
      if (!value.length) return 'No items';
      if (value.every(item => item === null || typeof item !== 'object')) return value.map(item => this.format(item)).join(', ');
      return `${value.length} items`;
    }
    const record = value as Record<string, unknown>;
    const name = [record['firstName'], record['lastName']].filter(item => typeof item === 'string' && item.trim()).join(' ');
    if (name) return name;
    const parts = Object.entries(record).filter(([, item]) => item !== null && typeof item !== 'object').slice(0, 2);
    return parts.length ? parts.map(([key, item]) => `${this.label(key)}: ${this.format(item)}`).join(' \u00b7 ') : `${Object.keys(record).length} fields`;
  }
  inspect(value: unknown, title: string): void {
    this.detailTrigger = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    if (this.selectedRecord === null) this.previousBodyOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    this.selectedRecord = value; this.detailTitle = title; this.detailFields = [];
    const flatten = (item: unknown, path: string) => {
      item = this.readableValue(item);
      if (!this.isStructured(item)) { this.detailFields.push({label: path || 'Value', value: this.format(item)}); return; }
      const entries = Object.entries(item as object);
      if (!entries.length) { this.detailFields.push({label: path || 'Value', value: Array.isArray(item) ? 'No items' : 'No fields'}); return; }
      for (const [key, child] of entries) {
        const label = Array.isArray(item) ? `Item ${Number(key) + 1}` : this.label(key);
        flatten(child, path ? `${path} / ${label}` : label);
      }
    };
    flatten(value, '');
    setTimeout(() => document.getElementById('close-details')?.focus());
  }
  detailKeydown(event: KeyboardEvent): void {
    if (event.key === 'Escape') { event.preventDefault(); this.closeDetails(); }
    if (event.key === 'Tab') { event.preventDefault(); document.getElementById('close-details')?.focus(); }
  }
  closeDetails(): void {
    if (this.selectedRecord === null) return;
    this.selectedRecord = null; document.body.style.overflow = this.previousBodyOverflow;
    if (this.detailTrigger?.isConnected) this.detailTrigger.focus();
  }
  chooseNo(): void { if (!this.busy && !this.feedbackSaved) this.feedbackChoice = 'no'; }

  async yes(): Promise<void> {
    if (this.busy || !this.requestId || this.feedbackSaved) return;
    this.feedbackSubmitting = true; this.error = '';
    try {
      const response = await firstValueFrom(this.api.submitFeedback(this.requestId, true));
      if (!response.feedback_saved) throw new Error('Feedback could not be saved. Please try again.');
      this.feedbackChoice = 'yes'; this.feedbackSaved = true; this.feedbackMessage = 'Thanks for your feedback.';
    } catch (error) { this.error = this.message(error); }
    finally { this.feedbackSubmitting = false; }
  }

  async submitNegative(): Promise<void> {
    if (this.busy || !this.requestId || !this.feedbackComment.trim() || this.feedbackSaved) return;
    this.feedbackSubmitting = true; this.error = ''; this.feedbackMessage = 'Processing feedback…';
    try {
      const response = await firstValueFrom(this.api.submitFeedback(this.requestId, false, this.feedbackComment.trim()));
      if (!response.feedback_saved) throw new Error('Feedback could not be saved. Please try again.');
      this.feedbackSaved = true;
      this.feedbackMessage = response.analysis_error ? 'Feedback saved; automatic learning was unavailable. Retrying the original question…' : 'Feedback processed. Regenerating answer…';
      this.runQuery(true);
    } catch (error) { this.error = this.message(error); this.feedbackMessage = 'Feedback was not saved. Please try again.'; }
    finally { this.feedbackSubmitting = false; }
  }
  retryRegeneration(): void { if (!this.busy && this.feedbackSaved && this.feedbackChoice === 'no') this.runQuery(true); }

  private message(error: unknown): string {
    if (error instanceof HttpErrorResponse) {
      if (error.status === 401) return 'API authentication is required. Configure the API key for this browser session.';
      if (error.status === 503) {
        this.schemaReady = false; this.healthStatus = 'Checking database readiness…';
        clearTimeout(this.healthTimer); this.checkHealth();
        return 'The backend is not ready. Please wait for schema preparation or check the database connection.';
      }
      if (error.status === 0) return 'The backend is unavailable. Check the connection and try again.';
      if (error.status === 422) return 'The query could not be validated. Refine the request and try again.';
      if (error.status === 404) return 'The answer history is unavailable. Ask the question again before submitting feedback.';
      return `The request failed (${error.status}). Please try again.`;
    }
    if (error instanceof Error && error.name === 'TimeoutError') return 'The query timed out. Please try again.';
    // Never display server exception text or raw stack traces.
    return 'The request could not be completed. Please try again.';
  }
}
