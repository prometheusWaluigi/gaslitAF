const fs = require('fs');
const path = require('path');

module.exports = function() {
  const baseDir = path.join(__dirname, '..', '..');
  const dirs = [
    'coreTheories',
    'simulationIdeas',
    'gettingWeird',
    'spikeProtein',
    'subTheories',
    'careManuals'
  ];

  const podcastCollections = [];
  const seenTitles = new Set();
  dirs.forEach(dir => {
    const mp3s = fs.readdirSync(path.join(baseDir, dir)).filter(f => f.endsWith('.mp3'));
    mp3s.forEach(file => {
      // Derive human-friendly title from filename (PREFIX_ Suffix)
      let title = file.replace('.mp3','');
      const match = title.match(/^(.+?)_\s*(.+)$/);
      if (match) {
        title = `${match[1]}: ${match[2]}`;
      }
      if (!seenTitles.has(title)) {
        podcastCollections.push({
          title,
          url: dir,
          file
        });
        seenTitles.add(title);
      }
    });
  });

  return { podcastCollections };
};
