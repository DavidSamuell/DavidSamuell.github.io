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
 "Hello! My name is David and welcome to my website. This website serve as my portfolio and I'll be posting some Data Science related project and content here. 

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