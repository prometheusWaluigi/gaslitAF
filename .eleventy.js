const fs = require('fs');
const path = require('path');
const markdownIt = require('markdown-it');
const markdownItAnchor = require('markdown-it-anchor');

module.exports = function(eleventyConfig) {
  // Enable Markdown-it with anchor support for headings
  const mdOptions = { html: true, breaks: true, linkify: true };
  const mdLib = markdownIt(mdOptions).use(markdownItAnchor, {
    permalink: markdownItAnchor.permalink.headerLink(),
    slugify: eleventyConfig.getFilter('slugify')
  });
  eleventyConfig.setLibrary('md', mdLib);

  // URL-encode filter
  eleventyConfig.addFilter('url_encode', str => encodeURIComponent(str));
  // Slugify titles for IDs
  eleventyConfig.addFilter('slugify', str => str.toString().toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, ''));

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
    pathPrefix: "/gaslitAF/",
    dir: {
      input: 'src',
      includes: '_includes',
      data: '_data',
      output: 'docs'
    }
  };
};
