---
layout: page
title: Scientific Publications
permalink: /publications/
---
See my profile on <a href="https://scholar.google.com/citations?user=Uona1HYAAAAJ">Google Scholar</a>.

Scientific *peer reviewed* publications:

[comment]: # (Had tp group publications to apply filter by year and title.)
{% assign sortedYears = site.data.publications | group_by:"Year" | sort:"Year" | reverse %}

<table class="sci-pub">
  {% for yearGroup in sortedYears %}
    {% if forloop.first %}
    <tr>
        <th>Title</th>
        <th>Year</th>
    </tr>
    {% endif %}
    {% assign sortedPubs = yearGroup.items | sort:"Title" %}
    {% for row in sortedPubs %}
      <tr>
          <td>
            <a href="{{ row["DoiLink"] }}" class="paper-titulo">
              {{ row["Title"] }}
            </a>
            <div class="autores">
              {{ row["Authors"] }}
            </div>
            <div class="autores">
              {{ row["Publication"] }}
            </div>
          </td>
          <td>{{ row["Year"]}}</td>
      </tr>
    {% endfor %}
  {% endfor %}
</table>
