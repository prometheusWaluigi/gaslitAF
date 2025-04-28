const fs = require('fs');
const path = require('path');

module.exports = function(eleventyConfig) {
  // Load and expose theory and podcast data globally
  const theoriesData = require('./src/_data/theories.js')();
  eleventyConfig.addGlobalData('theoryCollections', theoriesData.theoryCollections);
  const podcastsData = require('./src/_data/podcasts.js')();
  eleventyConfig.addGlobalData('podcastCollections', podcastsData.podcastCollections);

  // Custom filter to get current year
  eleventyConfig.addFilter('year', () => new Date().getFullYear());

  // Copy theory directories and podcast mp3s
  [
    'coreTheories',
    'simulationIdeas',
    'gettingWeird',
    'spikeProtein',
    'subTheories',
    'careManuals'
  ].forEach(dir => eleventyConfig.addPassthroughCopy(dir));
  // Copy global stylesheet
  eleventyConfig.addPassthroughCopy('styles.css');

  return {
    dir: {
      input: 'src',
      includes: '_includes',
      data: '_data',
      output: 'docs'
    }
  };
};
