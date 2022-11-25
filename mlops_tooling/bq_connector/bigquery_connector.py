from mlops_tooling.bq_connector.base_class import BaseConnector
from google.cloud import bigquery
from pathlib import Path
import re


CONFIG_DIR = (
    re.findall("^(\/[^\/]*\/[^\/]*\/)\/?", str(Path().absolute()))[0]
    + ".config/gcloud/prod.json"
)

SQL_DIR = str(Path().absolute().parent) + "/sql"


class BigQuery(BaseConnector):
    def __init__(self, path: str = SQL_DIR, credentials: str = CONFIG_DIR):
        self._credentials = credentials
        self._path = path
        super().__init__(path=self._path, credentials=self._credentials)

        self.Connector = self._instantiate_connection()

    def _instantiate_connection(self):
        return bigquery.Client.from_service_account_json(self._credentials)

    def _set_path(self):
        return super()._set_path()

    def _read_query(self, sql_file, **kwargs):
        return super()._read_query(sql_file, **kwargs)

    def query(self, sql_file, **kwargs):
        """
        Import a SQL script and output a dataframe of the results.

        Args:
            file_name: a filename of the SQL script.

        Returns:
            table: a pandas df of the output.
        """
        sql = self._read_query(sql_file, **kwargs)
        return self.Connector.query(sql).to_dataframe()
    
    def write_query(self, table_id, dataset, job_config):
        job = self.Connector.load_table_from_dataframe(
            dataset, table_id, job_config=job_config
        )
        job.result()
