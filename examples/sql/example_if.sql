select *
from example_table
{% if limit is defined %}
order by rand()
limit {{ limit }}
{% endif %}