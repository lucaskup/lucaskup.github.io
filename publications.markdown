---
layout: page
title: Scientific Publications
permalink: /publications/
---
<link rel="stylesheet" type="text/css" href="/assets/css/publication.css" />
See my profile on <a href="https://scholar.google.com/citations?user=Uona1HYAAAAJ">Google Scholar</a>.

Scientific *peer reviewed* publications:

[comment]: # (Had tp group publications to apply filter by year and title.)
{% assign sortedYears = site.data.publications | group_by:"Year" | sort:"Year" | reverse %}

<table>
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
            <a href="{{ row["DoiLink"] }}">
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
