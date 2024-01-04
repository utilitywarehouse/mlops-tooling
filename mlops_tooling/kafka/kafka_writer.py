from confluent_kafka import Producer
from datetime import datetime
from abc import ABC, abstractmethod

from mlops_tooling.kafka.any_pb import Any
from mlops_tooling.kafka.timestamp_pb import Timestamp
from mlops_tooling.kafka.errors import KafkaWriteError

import logging
import uuid
import pytz
import os


class KafkaWriter(ABC):
    def __init__(self, bootstrap_servers, topic):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.producer = Producer({"bootstrap.servers": self.bootstrap_servers})

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
        self.logger = logging.getLogger("KafkaWriter")

    @abstractmethod
    def pack_event(self):
        pass

    def write_message(self, event_class, package_class, sender_info, package, **kwargs):
        try:
            # Create an instance of the Event message
            event = event_class

            # Convert JSON to proto message
            event.id = str(uuid.uuid4())
            event = self.pack_event(event, **kwargs)

            # Get the current local time and convert to UTC
            local_time = datetime.now(pytz.timezone("Europe/London"))
            utc_time = local_time.astimezone(pytz.utc)

            # Create a Timestamp object and set its value to the UTC time
            timestamp = Timestamp()
            timestamp.FromDatetime(utc_time)

            event.timestamp.CopyFrom(timestamp)
            event.applies_at.CopyFrom(timestamp)

            # Set static values for sender and user fields
            sender = event.sender
            sender.domain = sender_info.domain
            sender.application = sender_info.application

            user = sender.user
            user.type = sender_info.type
            user.reference = sender_info.reference
            user.id = sender_info.id

            # Create package and set values
            package_payload = package_class
            for key, value in package.items():
                setattr(package_payload, key, value)

            payload = Any()
            payload.Pack(package_payload)
            event.payload.CopyFrom(payload)

            # Serialize the proto message to bytes
            serialized_message = event.SerializeToString()

            # Publish proto message to Kafka
            self.producer.produce(self.topic, value=serialized_message)

            self.producer.flush()

        except Exception as e:
            self.logger.error(e, exc_info=True)
            raise KafkaWriteError(repr(e))

    def close(self):
        self.producer.flush()
        self.producer.close()
