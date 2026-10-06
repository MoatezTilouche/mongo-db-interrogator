// Browser fixture verification: no MongoDB or OpenAI requests.
import { createServer } from 'node:http';
import { readFile, mkdir } from 'node:fs/promises';
import { resolve, extname } from 'node:path';
import assert from 'node:assert/strict';
import { chromium } from 'playwright';

const root = resolve('dist'), decisions = [];
let requestCount = 0, failExtraction = false;
const json = (res, status, value) => { res.writeHead(status, { 'Content-Type': 'application/json' }); res.end(JSON.stringify(value)); };
const server = createServer(async (req, res) => {
  const path = new URL(req.url, 'http://localhost').pathname;
  if (path === '/health') return json(res, 200, { status: 'ok', database: 'up', schema_ready: true });
  if (req.method === 'POST') {
    let text = ''; for await (const chunk of req) text += chunk;
    const body = JSON.parse(text);
    if (path.endsWith('/template-decision')) {
      decisions.push(body);
      if (failExtraction && body.action === 'reuse') { failExtraction = false; return json(res, 422, { detail: 'Could not safely extract parameters. Choose Regenerate.' }); }
      const reuse = body.action === 'reuse';
      return json(res, 200, { request_id: `req_${requestCount}`, status: 'completed', execution_success: true,
        semantic_template_cache: { checked: true, hit: true, user_decision: body.action },
        execution_result: [{ commandId: 'CMD-000009999999' }], final_response: reuse ? 'Reused successfully' : 'Regenerated successfully',
        validated_query: { operation: 'find', collection: 'orders', filter: { commandId: 'CMD-000009999999' } },
        execution_trace: (reuse ? ['TemplateDecision', 'TemplateParameterExtractor', 'QueryParameterBinder', 'ValidatorAgent', 'QueryCostGuard', 'ExecutorAgent'] : ['SemanticTemplateCache', 'SchemaRetriever', 'QueryContextAgent', 'QueryGeneratorAgent', 'ValidatorAgent', 'ExecutorAgent']).flatMap(agent => [{ agent, status: 'running' }, { agent, status: 'completed' }]) });
    }
    if (path === '/query' || path === '/query/stream') {
      requestCount++;
      const result = { request_id: `req_${requestCount}`, status: 'similar_template_detected', execution_success: false,
        similarity: { score: .91, template_id: 'tpl_fixture' },
        template_info: { source_question: 'Give me orders of command CMD-000009350132', semantic_intent: 'Retrieve orders for a command identifier',
          operation: 'find', collections: ['orders'], template: { operation: 'find', collection: 'orders', filter: { commandId: { $param: 'command_id' } } } },
        execution_trace: [{ agent: 'SemanticTemplateCache', status: 'completed', message: 'Similar template detected (91%)' }, { agent: 'TemplateDecision', status: 'pending', message: 'Waiting for user decision' }] };
      if (path === '/query') return json(res, 200, result);
      res.writeHead(200, { 'Content-Type': 'text/event-stream' });
      res.end(`event: result\ndata: ${JSON.stringify(result)}\n\nevent: done\ndata: {}\n\n`); return;
    }
  }
  try {
    const file = resolve(root, path === '/' ? 'index.html' : path.replace(/^\/static\//, ''));
    if (!file.startsWith(root + '\\') && !file.startsWith(root + '/')) return json(res, 404, {});
    const content = await readFile(file);
    res.writeHead(200, { 'Content-Type': { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css' }[extname(file)] || 'application/octet-stream' }); res.end(content);
  } catch { res.writeHead(404); res.end(); }
});
await new Promise(done => server.listen(0, '127.0.0.1', done));
let browser;
try {
  browser = await chromium.launch({ headless: true, executablePath: process.env.BROWSER_PATH || (process.platform === 'win32' ? 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe' : undefined) });
  const page = await browser.newPage(), errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  await page.getByText('Ready', { exact: true }).waitFor();
  const ask = async () => {
    await page.getByLabel('Your question').fill('Give me orders of command CMD-000009999999');
    await page.getByRole('button', { name: /^Ask/ }).click();
    await page.getByRole('heading', { name: 'The system detected a similar reusable template.' }).waitFor();
    assert.ok((await page.locator('.template-panel').textContent()).includes('91%'));
    assert.ok((await page.locator('.template-panel pre').textContent()).includes('$param'));
    assert.equal(await page.locator('#answer').count(), 0);
    assert.equal(await page.locator('.step').filter({ hasText: 'QueryGeneratorAgent' }).count(), 0);
    assert.ok(!(await page.locator('.template-panel').textContent()).includes('embedding'));
  };
  await ask();
  await page.setViewportSize({ width: 390, height: 844 });
  await mkdir('test-results', { recursive: true });
  await page.screenshot({ path: 'test-results/template-suggestion-mobile.png', fullPage: true });
  await page.getByRole('button', { name: 'Reuse template', exact: true }).click();
  await page.getByText('Reused successfully', { exact: true }).waitFor();
  assert.equal(await page.locator('.template-panel').count(), 0);
  assert.equal(await page.locator('.step').filter({ hasText: 'QueryContextAgent' }).count(), 0);
  assert.equal(await page.locator('.step').filter({ hasText: 'QueryGeneratorAgent' }).count(), 0);
  assert.equal(decisions[0].action, 'reuse');
  await ask(); failExtraction = true;
  await page.getByRole('button', { name: 'Reuse template', exact: true }).click();
  await page.getByText('The template could not be reused safely. Choose Regenerate.').waitFor();
  assert.equal(await page.locator('.template-panel').count(), 1);
  await page.getByRole('button', { name: 'Regenerate', exact: true }).click();
  await page.getByText('Regenerated successfully', { exact: true }).waitFor();
  assert.equal(decisions.at(-1).action, 'regenerate');
  assert.equal(await page.locator('.step').filter({ hasText: 'QueryGeneratorAgent' }).count(), 1);
  assert.deepEqual(errors, []);
  console.log('PASS: template suggestion, 91% similarity, JSON, mobile panel, explicit reuse/regenerate, extraction error/retry, no query agents on reuse, no embeddings, no browser errors.');
} finally { await browser?.close(); await new Promise(done => server.close(done)); }
