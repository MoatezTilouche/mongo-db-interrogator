// Browser verification against a local fixture API; no MongoDB/OpenAI requests.
import { createServer } from 'node:http';
import { mkdir, readFile } from 'node:fs/promises';
import { resolve, extname } from 'node:path';
import assert from 'node:assert/strict';
import { chromium } from 'playwright';

const root = resolve('dist');
let ready = false, queryCount = 0, failRegeneration = false, failFeedback = false;
let streamingAvailable = true;
const questions = [], feedbacks = [];
const json = (response, status, data) => { response.writeHead(status, { 'Content-Type': 'application/json' }); response.end(JSON.stringify(data)); };
const server = createServer(async (request, response) => {
  const path = new URL(request.url, 'http://localhost').pathname;
  if (path === '/health') return json(response, 200, { status: ready ? 'ok' : 'degraded', database: 'up', schema_ready: ready });
  if (request.method === 'POST') {
    let text = ''; for await (const chunk of request) text += chunk;
    const body = JSON.parse(text);
    if (path === '/feedback') {
      if (failFeedback) { failFeedback = false; return json(response, 503, { detail: 'fixture feedback failure' }); }
      feedbacks.push(body); return json(response, 200, { request_id: body.request_id, feedback_saved: true, analysis_created: !body.liked });
    }
    if (path === '/query/stream' && !streamingAvailable) return json(response, 404, {});
    if (path === '/query' || path === '/query/stream') {
      questions.push(body.question); queryCount++;
      if (failRegeneration) { failRegeneration = false; return json(response, 500, {}); }
      const result = {
        request_id: `req_${queryCount}`, execution_success: true,
        execution_result: [{ customer: '<script>unsafe</script>', total: queryCount * 100, optional: null, details: { firstName: 'Sample', lastName: 'Customer', addressInfo: { locality: 'Sfax', country: 'Tunisia' }, tags: ['test'] }, longText: 'A long note '.repeat(300) }, { customer: 'Bob', extra: 'dynamic column' }],
        final_response: `Summary ${queryCount}`, results_summary_agent_output: { summary_text: `Summary ${queryCount}` },
        schema_spec: {}, query_context: { operation: 'aggregate' }, generated_query: {}, validated_query: {},
        execution_path: { agents_called: ['query_context', 'query_generator', 'summary'], summary_skipped: false }
      };
      if (path === '/query') return json(response, 200, result);
      response.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' });
      const send = (event, data) => response.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
      const agents = ['Precheck', 'SchemaRetriever', 'QueryContextAgent', 'QueryGeneratorAgent', 'ValidatorAgent', 'ExecutorAgent', 'ResultsSummaryAgent'];
      for (const agent of agents) {
        send('workflow', { agent, status: 'running' });
        await new Promise(done => setTimeout(done, 120));
        send('workflow', { agent, status: 'completed', elapsed_ms: 120 });
      }
      send('result', result); send('done', {}); return response.end();
    }
  }
  try {
    const relative = path === '/' ? 'index.html' : path.replace(/^\/static\//, '');
    const file = resolve(root, relative);
    if (!file.startsWith(root + '\\') && !file.startsWith(root + '/')) return json(response, 404, {});
    const mime = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css' }[extname(file)] || 'application/octet-stream';
    response.writeHead(200, { 'Content-Type': mime }); response.end(await readFile(file));
  } catch { if (!response.headersSent) response.writeHead(404); response.end(); }
});
await new Promise(done => server.listen(0, '127.0.0.1', done));
const port = server.address().port;
let browser;
try {
  browser = await chromium.launch({ headless: true, executablePath: process.env.BROWSER_PATH || (process.platform === 'win32' ? 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe' : undefined) });
  const page = await browser.newPage();
  const errors = []; page.on('pageerror', error => errors.push(error.message));
  await page.goto(`http://127.0.0.1:${port}`);
  await page.getByRole('status').filter({ hasText: 'Preparing database schema...' }).waitFor();
  assert.equal(await page.getByRole('button', { name: /^Ask/ }).isEnabled(), false);
  assert.equal(await page.getByRole('heading', { name: 'Agent execution' }).count(), 0);
  assert.equal(await page.locator('#feedback').count(), 0);
  ready = true; await page.reload();
  await page.getByText('Ready', { exact: true }).waitFor();
  assert.equal(await page.getByRole('button', { name: /^Ask/ }).isEnabled(), false);
  await page.getByLabel('Your question').fill('Show top customers');
  await page.getByLabel('Your question').press('Control+Enter');
  await page.locator('.step.running').first().waitFor();
  await page.getByText('Summary 1', { exact: true }).waitFor();
  assert.equal(await page.locator('tbody tr').count(), 2);
  assert.equal(await page.getByRole('columnheader', { name: 'Extra', exact: true }).count(), 1);
  assert.equal(await page.locator('tbody tr').first().evaluate(row => row.getBoundingClientRect().height < 100), true);
  await page.getByRole('button', { name: /Sample Customer/ }).click();
  const dialog = page.getByRole('dialog');
  await dialog.waitFor();
  assert.equal(await dialog.getByText('Address Info / Locality', { exact: true }).count(), 1);
  assert.equal(await dialog.getByText('Sfax', { exact: true }).count(), 1);
  assert.equal(await dialog.getByText('Tags / Item 1', { exact: true }).count(), 1);
  await mkdir('test-results', { recursive: true });
  await page.screenshot({ path: 'test-results/record-details.png', fullPage: true });
  await page.getByRole('button', { name: 'Close', exact: true }).press('Tab');
  assert.equal(await page.getByRole('button', { name: 'Close', exact: true }).evaluate(button => button === document.activeElement), true);
  await page.keyboard.press('Escape');
  assert.equal(await dialog.count(), 0);
  await page.getByRole('button', { name: 'View record', exact: true }).first().click();
  assert.equal(await page.getByRole('dialog').getByText('Long Text', { exact: true }).count(), 1);
  await page.getByRole('button', { name: 'Close', exact: true }).click();
  assert.equal(await page.getByRole('cell', { name: 'null', exact: true }).count(), 1);
  await page.getByRole('button', { name: 'Yes', exact: true }).click();
  await page.getByText('Thanks for your feedback.').waitFor();
  assert.equal(queryCount, 1); assert.deepEqual(feedbacks[0], { request_id: 'req_1', liked: true });
  await page.getByLabel('Your question').fill('Show top customers');
  await page.getByRole('button', { name: /^Ask/ }).click();
  await page.getByText('Summary 2', { exact: true }).waitFor();
  await page.getByRole('button', { name: 'No', exact: true }).click();
  const regenerate = page.getByRole('button', { name: 'Submit feedback & regenerate', exact: true });
  assert.equal(await regenerate.isEnabled(), false);
  await page.getByLabel('What was wrong with the answer?').fill('   ');
  assert.equal(await regenerate.isEnabled(), false);
  await page.getByLabel('What was wrong with the answer?').fill('Rank by completed revenue');
  failFeedback = true; await regenerate.click();
  await page.getByText('Feedback was not saved. Please try again.').waitFor();
  assert.equal(queryCount, 2);
  failRegeneration = true; await regenerate.click();
  await page.getByRole('button', { name: 'Retry regeneration' }).waitFor();
  assert.equal(await page.getByText('Summary 2', { exact: true }).count(), 1);
  assert.deepEqual(feedbacks[1], { request_id: 'req_2', liked: false, comment: 'Rank by completed revenue' });
  await page.getByRole('button', { name: 'Retry regeneration' }).click();
  await page.getByText('Summary 4', { exact: true }).waitFor();
  assert.equal(feedbacks.length, 2);
  assert.ok(questions.every(question => question === 'Show top customers'));
  await page.getByRole('button', { name: 'Yes', exact: true }).click();
  await page.getByText('Thanks for your feedback.').waitFor();
  assert.equal(feedbacks[2].request_id, 'req_4');
  streamingAvailable = false;
  await page.getByRole('button', { name: /^Ask/ }).click();
  await page.getByText('Summary 5', { exact: true }).waitFor();
  await page.getByRole('button', { name: 'Show steps', exact: true }).click();
  assert.equal(await page.locator('.step.completed').count(), 7);
  await mkdir('test-results', { recursive: true });
  await page.screenshot({ path: 'test-results/angular-desktop.png', fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  assert.equal(await page.locator('.table-wrap').evaluate(element => element.scrollWidth > element.clientWidth), true);
  await page.screenshot({ path: 'test-results/angular-mobile.png', fullPage: true });
  assert.deepEqual(errors, []);
  if (process.env.ANGULAR_DEV_URL) {
    const devPage = await browser.newPage();
    devPage.on('pageerror', error => errors.push(error.message));
    await devPage.route('**/health', route => route.fulfill({ json: { status: 'ok', database: 'up', schema_ready: true } }));
    await devPage.goto(process.env.ANGULAR_DEV_URL);
    await devPage.getByText('Ready', { exact: true }).waitFor();
    await devPage.getByLabel('Your question').fill('A development question');
    assert.equal(await devPage.getByRole('button', { name: /^Ask/ }).isEnabled(), true);
    assert.deepEqual(errors, []);
    await devPage.close();
    console.log('PASS: Angular development server page and assets load.');
  }
  console.log('PASS: readiness, live SSE, dynamic results, summary, Yes, required No comment, feedback failure, regeneration failure/retry, original question, request IDs, /query fallback, mobile layout, no browser errors.');
} finally { await browser?.close(); server.closeAllConnections(); await new Promise(done => server.close(done)); }
