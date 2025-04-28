---
layout: default
title: GASLIT-AF Podcast & Theories
---

# Welcome to GASLIT-AF

Explore our comprehensive repository of theoretical papers and listen to our podcast series.

## Theories

<ul>
  {% for dir in theoryCollections %}
    <li><a href="{{ dir.url }}index.html">{{ dir.title }}</a></li>
  {% endfor %}
</ul>

## Podcasts

<ul>
  {% for podcast in podcastCollections %}
    <li id="{{ podcast.title | slugify }}">
      <strong>{{ podcast.title }}</strong><br>
      <audio controls id="audio-{{ podcast.title | slugify }}">
        <source src="{{ podcast.url }}{{ podcast.file }}" type="audio/mpeg">
        Your browser does not support the audio element.
      </audio>
    </li>
  {% endfor %}
</ul>
