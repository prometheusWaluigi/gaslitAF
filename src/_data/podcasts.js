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
      const title = file.replace('.mp3','');
      if (!seenTitles.has(title)) {
        podcastCollections.push({
          title,
          url: `/${dir}/`,
          file
        });
        seenTitles.add(title);
      }
    });
  });

  return { podcastCollections };
};
