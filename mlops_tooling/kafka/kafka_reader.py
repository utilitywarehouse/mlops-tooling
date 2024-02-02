from confluent_kafka import Consumer, KafkaError, TopicPartition

from mlops_tooling.kafka.errors import KafkaReadError

import logging
import os
import time


class KafkaReader:
    def __init__(
        self, bootstrap_servers, topic, group_id=None, auto_offset_reset="latest"
    ):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.auto_offset_reset = auto_offset_reset
        self.consumer = None

        # Configure logging
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s"
        formatter = logging.Formatter(log_format)

        # Get the desired logging level from the environment
        log_level = os.environ.get("LOG_LEVEL", "INFO")

        # Convert the logging level from string to the corresponding constant
        log_level = getattr(logging, log_level.upper())

        # Configure logging
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        root_logger.handlers = []
        root_logger.addHandler(logging.StreamHandler())
        root_logger.handlers[0].setFormatter(formatter)

        # Create a logger for the KafkaReader class
        self.logger = logging.getLogger("KafkaReader")

    def connect(self):
        self.consumer = Consumer(
            {
                "bootstrap.servers": self.bootstrap_servers,
                "group.id": self.group_id,
                "auto.offset.reset": self.auto_offset_reset,
                "enable.auto.commit": False,
            }
        )
        self.consumer.subscribe([self.topic])

    def read_messages(self, max_retries=3, retry_delay=1):
        retries = 0
        # Keep track of assigned partitions
        assigned_partitions = set()

        while retries < max_retries:
            try:
                if not assigned_partitions:
                    # Fetch assigned partitions only if there are no partitions assigned
                    assigned_partitions = self.consumer.assignment()

                message = self.consumer.poll(timeout=1.0)

                if message is None:
                    self.logger.info("No new messages. Sleeping for a while...")
                    time.sleep(retry_delay)

                elif not message.error():
                    return message

                else:
                    error = message.error()

                    if error.code() == KafkaError._PARTITION_EOF:
                        self.logger.info("Reached end of partition. Rebalancing...")
                        assigned_partitions = set()  # Clear assigned partitions
                        time.sleep(retry_delay)

                    else:
                        self.logger.error(f"Error occurred: {error}", exc_info=True)
                        retries += 1
                        time.sleep(retry_delay)

            except Exception as e:
                self.logger.error(f"Error occurred: {e}", exc_info=True)
                retries += 1
                time.sleep(retry_delay)

        self.logger.error(
            "Reached maximum retries. Unable to read messages.", exc_info=True
        )
        raise KafkaReadError("Reached maximum retries. Unable to read messages.")

    def commit_with_retries(self, message, max_retries=3, retry_delay=1):
        retries = 0
        while retries < max_retries:
            try:
                self.consumer.commit(message)

            except Exception as e:
                self.logger.error(f"Commit failed: {e}. Retrying...", exc_info=True)
                retries += 1
                time.sleep(retry_delay)

        self.logger.error(
            f"Reached maximum retries. Unable to commit offset for message {message}"
        )

        raise KafkaReadError(
            f"Reached maximum retries. Unable to commit offset for message {message}",
        )

    def close(self):
        if self.consumer is not None:
            self.consumer.close()
