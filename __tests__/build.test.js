const fs = require('fs');
const path = require('path');

// Ensure Eleventy has built the site before running tests
beforeAll(() => {
  // Build the site
  require('child_process').execSync('npm run build', { cwd: process.cwd(), stdio: 'inherit' });
});

describe('Eleventy Build', () => {
  test('docs/index.html should exist', () => {
    const indexPath = path.join(process.cwd(), 'docs', 'index.html');
    expect(fs.existsSync(indexPath)).toBe(true);
  });

  test('docs directory should contain theory folders', () => {
    const theoryDir = path.join(process.cwd(), 'docs', 'coreTheories');
    expect(fs.existsSync(theoryDir)).toBe(true);
  });

  test('docs directory should contain mp3 files', () => {
    const mp3Path = path.join(process.cwd(), 'docs', 'coreTheories');
    const files = fs.readdirSync(mp3Path).filter(f => f.endsWith('.mp3'));
    expect(files.length).toBeGreaterThan(0);
  });
});
