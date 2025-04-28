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

  const theoryCollections = dirs.map(dir => {
    const dirPath = path.join(baseDir, dir);
    const files = fs.readdirSync(dirPath).filter(f => f.endsWith('.md'));
    return {
      title: dir,
      url: dir,
      files
    };
  });

  return { theoryCollections };
};
