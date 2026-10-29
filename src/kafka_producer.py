
import csv
import json
import time
import argparse
from pathlib import Path

from kafka import KafkaProducer
from kafka.errors import KafkaError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "AirQuality_Model_Input.csv"

BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC = "air-quality"


def main():
    parser = argparse.ArgumentParser(
        description="Stream historical air-quality records to Kafka."
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between records (default: 1)"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum records to publish (default: 20; use 0 for all)"
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based CSV data-row index to start from (default: 0)"
    )

    args = parser.parse_args()

    if args.interval < 0 or args.limit < 0 or args.start_index < 0:
        raise ValueError(
            "Interval, limit, and start-index must be non-negative."
        )

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        acks="all",
        retries=3
    )

    sent = 0

    try:
        with DATA_FILE.open(
            "r", newline="", encoding="utf-8-sig"
        ) as file:
            reader = csv.DictReader(file)

            # Skip the requested number of data rows.
            for _ in range(args.start_index):
                try:
                    next(reader)
                except StopIteration:
                    print(
                        f"Start index {args.start_index} is beyond "
                        "the available dataset rows."
                    )
                    return

            for row in reader:
                if args.limit and sent >= args.limit:
                    break

                # Convert numeric measurements to numbers for JSON.
                event = {}

                for key, value in row.items():
                    if value is None or value.strip() == "":
                        event[key] = None

                    elif key == "Timestamp":
                        event[key] = value

                    else:
                        try:
                            event[key] = float(value)
                        except ValueError:
                            event[key] = value

                future = producer.send(TOPIC, value=event)
                future.get(timeout=30)

                sent += 1

                print(
                    f"Published record {sent} "
                    f"(CSV data row {args.start_index + sent}): "
                    f"Timestamp={event.get('Timestamp')}, "
                    f"Target_Next_Hour={event.get('Target_Next_Hour')}"
                )

                if args.interval:
                    time.sleep(args.interval)

        producer.flush()

        print(
            f"\nFinished. Published {sent} records "
            f"to topic '{TOPIC}'."
        )

    except (KafkaError, OSError, ValueError) as error:
        print(f"Producer failed: {error}")
        raise

    finally:
        producer.close()


if __name__ == "__main__":
    main()
