select
  {{ id_column }}
  , {{ column_1 }}
  , {{ column_2 }}
from {{ table_name }}
{% if sample is defined %}
tablesample system ({{ sample }} percent)
{% endif %}
{% if limit is defined %}
order by rand()
limit {{ limit }}
{% endif %}
