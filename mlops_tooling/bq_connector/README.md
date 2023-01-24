## BigQuery connector

BigQuery connector uses Jinja2 to enable the user to write .sql files and import them into their notebooks. We can the connector by creating a bq class specifying the path to your sql folder, and the location of your credentials.

```python
bq = BigQuery(path = "path_to_sql_files", credentials = "ceredentials/path")
data = bq.query("example_query.sql")
```
