from flwr.clientapp import ClientApp

from flwr.app import (
    Context,
    Message,
    ArrayRecord,
    MetricRecord,
    RecordDict,
)

from tensorflow.keras import backend as K

from plant_fl.task import (
    load_model,
    load_data,
)

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):

    # Clear TensorFlow session
    K.clear_session()

    # ------------------------------------------------
    # Which client is this?
    # ------------------------------------------------

    client_name = context.node_config["client-name"]

    # ------------------------------------------------
    # Load local dataset
    # ------------------------------------------------

    dataset, _ = load_data(client_name)

    # ------------------------------------------------
    # Build model
    # ------------------------------------------------

    learning_rate = context.run_config["learning-rate"]

    model = load_model(learning_rate)

    # ------------------------------------------------
    # Load global weights
    # ------------------------------------------------

    model.set_weights(

        msg.content["arrays"].to_numpy_ndarrays()

    )

    # ------------------------------------------------
    # Local training
    # ------------------------------------------------

    epochs = context.run_config["local-epochs"]

    history = model.fit(

        dataset,

        epochs=epochs,

        verbose=1

    )

    # ------------------------------------------------
    # Metrics
    # ------------------------------------------------

    train_loss = history.history["loss"][-1]

    train_accuracy = history.history["accuracy"][-1]

    metrics = MetricRecord(

        {

            "train_loss": train_loss,

            "train_accuracy": train_accuracy,

            "num-examples": len(dataset)

        }

    )

    # ------------------------------------------------
    # Updated model
    # ------------------------------------------------

    arrays = ArrayRecord(

        model.get_weights()

    )

    content = RecordDict(

        {

            "arrays": arrays,

            "metrics": metrics

        }

    )

    return Message(

        content=content,

        reply_to=msg

    )