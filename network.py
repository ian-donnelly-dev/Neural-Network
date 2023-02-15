def predict(network, value):
    for layer in network:
        value = layer.forward(value)

    return value

def train(network, loss, loss_prime, x_train, y_train, epochs, learning_rate):
    for epoch in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            output = predict(network, x)

            error += loss(y, output)

            gradient = loss_prime(y, output)
            for layer in reversed(network):
                gradient = layer.backward(gradient, learning_rate)

        error /= len(x_train)

        print(f"epoch = {(epoch + 1):0{len(str(epochs))}d}/{epochs} | error = {error:.10f}")