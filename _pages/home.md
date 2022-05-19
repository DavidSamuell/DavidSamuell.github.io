---
permalink: /
title: " "
excerpt:
author_profile: true
layout: single
header:
    overlay_image: /header/home-header.jpg
    overlay_filter: 0.5
---


<div style="margin-bottom: 2em">
 Hello! My name is <b>David</b> and welcome to my website. I'm a Computer Science Undergraduate at Binus University. I have a strong passion for the field of Machine and Deep Learning and am highly self motivated to constantly improve my skills and knowledge in the field. I like to work on projects on my sparetime and I believe that it's the best way to learn something new. I hope that you enjoy the articles I posted, cheers✌️.
 </div>


## Recent Post
{% if paginator %}
  {% assign posts = paginator.portfolio | sort: 'date' | reverse %}
{% else %}
  {% assign posts = site.portfolio | sort: 'date' | reverse %}
{% endif %}

{% assign entries_layout = page.entries_layout | default: 'list'  %}
<div class="entries-{{ entries_layout }}">
  {% for post in posts limit: 3 %}
    {% include archive-single.html type=entries_layout %}
  {% endfor %}
</div>

{% include paginator.html %}