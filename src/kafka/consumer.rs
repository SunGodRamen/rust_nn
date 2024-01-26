use rdkafka::config::ClientConfig;
use rdkafka::consumer::{Consumer, StreamConsumer};
use rdkafka::Message;
use crate::kafka::message::KafkaMessage;
use serde_json;
use config::{Config, File};

pub async fn start_kafka_consumer(network: &mut NeuralNetwork) {
    // Load configuration
    let settings = Config::builder()
        .add_source(File::with_name("config.toml"))
        .build()
        .expect("Configuration file loading failed");

    let group_id = settings.get_string("group.id").unwrap();
    let bootstrap_servers = settings.get_string("bootstrap.servers").unwrap();
    let enable_auto_commit = settings.get_string("commit.enable_auto_commit").unwrap();
    let auto_offset_reset = settings.get_string("offset.auto_offset_reset").unwrap();
    let topic_name = settings.get_string("topic.name").unwrap();

    // Configure Kafka consumer with settings from config file
    let consumer: StreamConsumer = ClientConfig::new()
        .set("group.id", &group_id)
        .set("bootstrap.servers", &bootstrap_servers)
        .set("enable.auto.commit", &enable_auto_commit)
        .set("auto.offset.reset", &auto_offset_reset)
        .create()
        .expect("Consumer creation failed");

    consumer.subscribe(&[&topic_name])
        .expect("Can't subscribe to specified topic");

    let mut network = NeuralNetwork::new(&[...]); // Initialize your network

    loop {
        match consumer.poll(None) {
            Some(Ok(message)) => {
                match message.payload_view::<str>() {
                    Some(Ok(payload)) => {
                        let input = deserialize_payload(payload);
                        let output = network.forward(&input);
                        println!("Processed Output: {:?}", output);
                    },
                    _ => println!("Error while reading message payload"),
                }
            },
            _ => (),
        }
    }
}

fn deserialize_payload(payload: &str) -> Vec<f64> {
    // Deserialize the JSON string into KafkaMessage
    let message: KafkaMessage = serde_json::from_str(payload)
        .expect("Failed to deserialize payload");

    // Convert the message to a format suitable for the neural network
    // For example, here we're just using num_windows and cursor position x and y
    vec![message.num_windows as f64, message.cursor_position.x as f64, message.cursor_position.y as f64]
}