import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpDownloadProgressEvent, HttpEventType, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';

export type AgentStatus = 'pending' | 'running' | 'completed' | 'failed' | 'skipped';
export interface WorkflowEvent {
  agent: string;
  status: AgentStatus;
  elapsed_ms?: number;
  message?: string;
}
export interface QueryResponse {
  request_id: string;
  status?: string;
  question?: string | null;
  optimization_iteration?: number;
  optimization_history?: Record<string, unknown>[];
  optimization?: Record<string, unknown>;
  message?: string;
  cost_analysis?: { estimated_rows?: number; risk_score?: number; signals?: string[] };
  suggestions?: { id: string; label: string; type?: string; value?: number }[];
  execution_trace?: WorkflowEvent[];
  pagination?: { page: number; page_size: number; has_next: boolean; has_previous: boolean };
  execution_success: boolean;
  execution_result?: Record<string, unknown>[] | null;
  execution_error?: string | null;
  execution_error_code?: string | null;
  final_response?: string | null;
  results_summary_agent_output?: { summary_text?: string } | null;
  execution_path?: { agents_called?: string[]; summary_skipped?: boolean } | null;
  query_context?: Record<string, unknown> | null;
  schema_spec?: Record<string, unknown> | null;
  generated_query?: Record<string, unknown> | null;
  validated_query?: Record<string, unknown> | null;
  execution_metadata?: { row_count?: number; execution_time_ms?: number } | null;
}
export interface FeedbackResponse {
  request_id: string;
  feedback_saved: boolean;
  analysis_created: boolean;
  analysis_error?: string | null;
}
export interface HealthResponse { status: string; database: string; schema_ready: boolean; }
export type QueryEvent = { type: 'workflow'; data: WorkflowEvent } | { type: 'result'; data: QueryResponse };

@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = environment.apiBaseUrl.replace(/\/$/, '');

  private headers(): HttpHeaders {
    // Optional shared-key support without an authentication screen or bundled secret.
    const key = sessionStorage.getItem('mongo-api-key');
    return key ? new HttpHeaders({ 'X-API-Key': key }) : new HttpHeaders();
  }

  query(question: string): Observable<QueryResponse> {
    return this.http.post<QueryResponse>(`${this.baseUrl}/query`, { question }, { headers: this.headers() });
  }

  optimizeQuery(requestId: string, payload: { optimization: string; value?: number; page?: number; refinement?: string }): Observable<QueryResponse> {
    return this.http.post<QueryResponse>(`${this.baseUrl}/query/${encodeURIComponent(requestId)}/optimize`, payload, { headers: this.headers() });
  }

  submitFeedback(requestId: string, liked: boolean, comment?: string): Observable<FeedbackResponse> {
    const payload = comment === undefined ? { request_id: requestId, liked } : { request_id: requestId, liked, comment };
    return this.http.post<FeedbackResponse>(`${this.baseUrl}/feedback`, payload, { headers: this.headers() });
  }

  health(): Observable<HealthResponse> {
    return this.http.get<HealthResponse>(`${this.baseUrl}/health`, { timeout: 10000 });
  }

  streamQuery(question: string): Observable<QueryEvent> {
    return new Observable(subscriber => {
      let consumed = 0, receivedResult = false;
      const parse = (text: string) => {
        let boundary: number;
        while ((boundary = text.indexOf('\n\n', consumed)) >= 0) {
          const frame = text.slice(consumed, boundary); consumed = boundary + 2;
          const lines = frame.split('\n');
          const event = lines.find(line => line.startsWith('event: '))?.slice(7);
          const data = lines.filter(line => line.startsWith('data: ')).map(line => line.slice(6)).join('\n');
          if (!data) continue;
          const payload = JSON.parse(data);
          if (event === 'workflow') subscriber.next({ type: 'workflow', data: payload });
          else if (event === 'result') { receivedResult = true; subscriber.next({ type: 'result', data: payload }); }
          else if (event === 'error') throw new Error(payload.message || 'Query could not be completed.');
        }
      };
      const subscription = this.http.post(`${this.baseUrl}/query/stream`, { question }, {
        headers: this.headers(), responseType: 'text', observe: 'events', reportProgress: true
      }).subscribe({
        next: event => {
          try {
            if (event.type === HttpEventType.DownloadProgress) parse((event as HttpDownloadProgressEvent).partialText || '');
            else if (event.type === HttpEventType.Response) parse(event.body || '');
          } catch (error) { subscriber.error(error); }
        },
        error: error => subscriber.error(error),
        complete: () => receivedResult ? subscriber.complete() : subscriber.error(new Error('Connection ended before an answer arrived. Please try again.'))
      });
      return () => subscription.unsubscribe();
    });
  }
}
