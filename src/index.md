---
layout: default
title: GASLIT-AF Podcast & Theories
---

# Welcome to GASLIT-AF

Explore our comprehensive repository of theoretical papers and listen to our podcast series.

## Theories

<ul>
  {% for dir in collections.theoryCollections %}
  <li><a href="/{{ dir.url }}index.html">{{ dir.data.title }}</a></li>
  {% endfor %}
</ul>

## Podcasts

<ul>
  {% for podcast in collections.podcastCollections %}
  <li>
    <strong>{{ podcast.data.title }}</strong><br>
    <audio controls>
      <source src="/{{ podcast.url }}{{ podcast.data.file }}" type="audio/mpeg">
      Your browser does not support the audio element.
    </audio>
  </li>
  {% endfor %}
</ul>
