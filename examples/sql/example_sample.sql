select *
from example_table
{% if sample is defined %}
tablesample system ({{ sample }} percent)
{% endif %}