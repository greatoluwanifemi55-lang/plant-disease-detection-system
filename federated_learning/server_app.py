from flwr.serverapp import ServerApp

from flwr.server.strategy import FedAvg

app = ServerApp()


@app.main()
def main(grid, context):

    strategy = FedAvg(

        fraction_train=1.0,

        fraction_evaluate=0.0,

        min_available_nodes=3,

        min_train_nodes=3

    )

    strategy.start(
        grid=grid,
        initial_arrays=None
    )