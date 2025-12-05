import re
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

from mlops_tooling.bq_connector.base_class import BaseConnector

try:
    CONFIG_DIR = (
        re.findall("^(\/[^\/]*\/[^\/]*\/)\/?", str(Path().absolute()))[0]
        + ".config/gcloud/prod.json"
    )

except Exception:
    CONFIG_DIR = "./"

try:
    SQL_DIR = str(Path().absolute().parent) + "/sql"

except Exception:
    SQL_DIR = "./sql"


class BigQuery(BaseConnector):
    def __init__(self, path: str = SQL_DIR, credentials: str = CONFIG_DIR):
        self._credentials = credentials
        self._path = path
        super().__init__(path=self._path, credentials=self._credentials)

        self.Connector = self._instantiate_connection()

    def _instantiate_connection(self):
        """
        Creates a connection to the BigQuery database.
        """
        return bigquery.Client.from_service_account_json(self._credentials)

    def _set_path(self):
        """
        Sets the path to the SQL files
        """
        return super()._set_path()

    def _read_query(self, sql_file, **kwargs):
        """
        Imports a SQL script and reads the file.

        Parameters
        ----------
        file_name : str
            A filename of the SQL script.

        Returns
        ----------
        query : str
            A string of the SQL query.
        """
        return super()._read_query(sql_file, **kwargs)

    def table(self, table_name: str):
        """
        Returns all values from a table and outputs a dataframe of the results.

        Parameters
        ----------
        table_name : str
            BigQuery table name.

        Returns
        ----------
        table: pd.DataFrame
            A pandas df of the output.
        """
        sql = f"select * from {table_name}"
        return self.Connector.query(sql).to_dataframe()

    def query(self, sql_file: str, create_bqstorage_client: bool = False, **kwargs):
        """
        Import a SQL script and output a dataframe of the results.

        Parameters
        ----------
        sql_file : str
            A filename of the SQL script.

        Returns
        ----------
        table: pd.DataFrame
            A pandas df of the output.
        """
        sql = self._read_query(sql_file, **kwargs)
        return self.Connector.query(sql).to_dataframe(
            create_bqstorage_client=create_bqstorage_client
        )

    def query_in_chunks(
        self,
        sql_file: str,
        chunk_size: int = 10000,
        query_timeout: int = 3600,
        result_timeout: int = 3600,
        schema: list[str] = [],
        **kwargs,
    ):
        """
        Returns all values from a table and outputs results in chunks.

        Parameters
        ----------
        sql_file : str
            A filename of the SQL script.
        chunk_size : int
            Size of the chunks to return.
        schema : list[str]
            Columns the query will return.

        Returns
        ----------
        table: pd.DataFrame or list[Row]
            A pandas df of the output.
        """
        sql = self._read_query(sql_file, **kwargs)
        query_job = self.Connector.query(sql, timeout=query_timeout)

        results = query_job.result(page_size=chunk_size, timeout=result_timeout)

        for page in results.pages:
            if schema:
                yield pd.DataFrame([row.values() for row in page], columns=schema)

            else:
                yield page

    def write_query(self, table_id, dataset, job_config):
        """
        Create a new table in BigQuery containing data from a Pandas DataFrame.

        Parameters
        ----------
        table_id : str
            The name of the table to create/add to.
        dataset : pd.DataFrame
            The dataset to add the table to.
        job_config : dict
            A dictionary containing the configuration for the job. See Google Job Config documentation.
        """
        job = self.Connector.load_table_from_dataframe(
            dataset, table_id, job_config=job_config
        )
        job.result()
