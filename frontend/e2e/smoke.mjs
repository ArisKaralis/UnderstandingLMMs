// Headless click-through of every page and trace stage; fails on any page error.
// usage: node e2e/smoke.mjs [baseUrl] [screenshotDir]
import { chromium } from "playwright";
const base = process.argv[2] || "http://localhost:8000";
const shots = process.argv[3] || null;
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1500, height: 1000 } });
const errors = [];
page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
page.on("console", (m) => { if (m.type() === "error") errors.push(`console: ${m.text()}`); });
const snap = async (name) => { if (shots) await page.screenshot({ path: `${shots}/${name}.png`, fullPage: true }); };
const wait = (ms) => page.waitForTimeout(ms);

await page.goto(`${base}/specs`); await page.waitForSelector("table"); await wait(500); await snap("specs");
await page.goto(`${base}/architecture`); await page.waitForSelector(".cell"); await wait(500);
await page.locator(".cell.attn").first().click(); await wait(300);
await page.locator(".macro .node").last().click(); await wait(300);
for (const d of await page.locator("details.disclosure summary").all()) await d.click();
await wait(400); await snap("architecture");
await page.goto(`${base}/trace`); await page.waitForSelector(".stagenav"); await wait(1500);
const stages = await page.locator(".stagenav button").count();
for (let i = 0; i < Math.min(7, stages); i++) {
  await page.locator(".stagenav button").nth(i).click(); await wait(2500); await snap(`trace_${i}`);
}
for (const p of ["experiments", "training", "learn"]) { await page.goto(`${base}/${p}`); await wait(1200); await snap(p); }
await page.goto(`${base}/training`); await wait(500);
for (let i = 1; i < 5; i++) { await page.locator(".stagenav button").nth(i).click(); await wait(1200); await snap(`training_${i}`); }
await browser.close();
if (errors.length) { console.error(errors.join("\n")); process.exit(1); }
console.log("E2E SMOKE PASSED");
