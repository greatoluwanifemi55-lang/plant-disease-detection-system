import numpy as np


class FederatedServer:

    def aggregate(self, clients):
        """
        Weighted Federated Averaging (FedAvg).
        """

        total_examples = sum(client.num_examples for client in clients)

        client_weights = [client.get_weights() for client in clients]

        averaged_weights = []

        for layer_weights in zip(*client_weights):

            weighted_layer = np.zeros_like(layer_weights[0])

            for client, weights in zip(clients, layer_weights):
                weighted_layer += (
                    weights * client.num_examples / total_examples
                )

            averaged_weights.append(weighted_layer)

        return averaged_weights