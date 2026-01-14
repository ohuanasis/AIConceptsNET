namespace NeuralNetwork
{
    internal class Program
    {
        static void Main(string[] args)
        {
            // Create a tiny neural network:
            // - 3 inputs (features)
            // - 1 output (a single prediction value between 0 and 1)
            // - weights learned by training
            NeuralNetwork neuralNetwork = new NeuralNetwork();

            // Training data:
            // Each row is one training example with 3 input features.
            // (This is a very small dataset just to demonstrate the math.)
            double[,] trainingSetInputs = new double[,]
            {
                {0, 0, 0},
                {1, 1, 1},
                {1, 0, 0}
            };

            // Expected outputs (labels) for each input row above.
            // One output value per training example.
            double[,] trainingSetOutputs = new double[,]
            {
                {0},
                {1},
                {1}
            };

            // Train for N iterations:
            // Repeatedly:
            //  1) run a forward pass to get predictions
            //  2) compute error (target - prediction)
            //  3) compute how to adjust weights (gradient step)
            //  4) update weights
            neuralNetwork.Train(trainingSetInputs, trainingSetOutputs, 1000);

            // After training, run predictions on new inputs.
            // Each row is a new example to classify/predict.
            double[,] output = neuralNetwork.Think(new double[,]
            {
                {0,1,0},
                {0,0,0},
                {0,0,1}
            });

            // Print the results. (Rounding here makes it look like 0/1 classification.)
            PrintMatrix(output);
        }

        private static void PrintMatrix(double[,] matrix)
        {
            // Utility to print a 2D array (matrix) row by row.
            int rows = matrix.GetLength(0);
            int columns = matrix.GetLength(1);

            for (int row = 0; row < rows; row++)
            {
                for (int column = 0; column < columns; column++)
                {
                    // Rounding is just presentation:
                    // the network outputs values in (0, 1) due to sigmoid.
                    Console.Write(Math.Round(matrix[row, column]) + "\t");
                }
                Console.WriteLine();
            }
        }
    }

    internal class NeuralNetwork
    {
        // Weight matrix:
        // Shape = [numberOfInputNodes x numberOfOutputNodes]
        // Here: [3 x 1]
        //
        // During Think():
        //   output = sigmoid(inputs • weights)
        // where "•" is matrix dot-product.
        private double[,] weights;

        // Supported element-wise operations for matrices of the same shape.
        enum OPERATION { Multiply, Add, Subtract }

        public NeuralNetwork()
        {
            // Seeded RNG so runs are deterministic (same initial weights each run).
            Random randomNumber = new Random(1);

            int numerOfInputNodes = 3;
            int numberOfOutputNodes = 1;

            // Initialize weights randomly.
            // (In real ML you usually pick small values centered around 0,
            // but this is fine for the demo.)
            weights = new double[numerOfInputNodes, numberOfOutputNodes];

            for (int i = 0; i < numerOfInputNodes; i++)
            {
                for (int j = 0; j < numberOfOutputNodes; j++)
                {
                    weights[i, j] = randomNumber.NextDouble();
                }
            }
        }

        private double[,] Activate(double[,] matrix, bool IsDerivative)
        {
            // This method applies the sigmoid activation function element-wise:
            //    sigmoid(x) = 1 / (1 + e^(-x))
            //
            // If IsDerivative == true, it returns the derivative used for training
            // (i.e., for gradient calculation).
            //
            // IMPORTANT NOTE (about this code as written):
            // The derivative line uses: matrix[row, column] * (1 - matrix[row, column])
            // which is the derivative in terms of sigmoid output (a), not the raw x.
            // That means the function expects "matrix" to already be sigmoid(output)
            // when IsDerivative == true (and in Train() you do pass "output").
            int numerOfRows = matrix.GetLength(0);
            int numberOfColumns = matrix.GetLength(1);

            double[,] result = new double[numerOfRows, numberOfColumns];

            for (int row = 0; row < numerOfRows; row++)
            {
                for (int column = 0; column < numberOfColumns; column++)
                {
                    // Forward activation: sigmoid(x)
                    double sigmoidOutput =
                        result[row, column] = 1 / (1 + Math.Exp(-matrix[row, column]));

                    // Derivative of sigmoid in terms of its output "a":
                    // sigmoid'(a) = a * (1 - a)
                    //
                    // In the training flow, "matrix" is the already-activated output,
                    // so this derivative uses that value.
                    double derivativeSigmoidOutput =
                        result[row, column] = matrix[row, column] * (1 - matrix[row, column]);

                    // Choose which value to return based on caller intent.
                    result[row, column] = IsDerivative ? derivativeSigmoidOutput : sigmoidOutput;
                }
            }

            return result;
        }

        public void Train(double[,] trainingInputs, double[,] trainingOutputs, int numberOfIterations)
        {
            // Training loop uses a simple gradient-descent-like update for a single-layer network.
            //
            // This is effectively logistic regression / a single-layer perceptron with sigmoid,
            // trained via gradient steps derived from the error.
            for (int iteration = 0; iteration < numberOfIterations; iteration++)
            {
                // 1) Forward pass: run the network on the training inputs.
                // output shape: [numSamples x 1]
                double[,] output = Think(trainingInputs);

                // 2) Compute error:
                // error = target - prediction
                // same shape as output: [numSamples x 1]
                double[,] error = PerformOperation(trainingOutputs, output, OPERATION.Subtract);

                // 3) Compute how much to adjust weights:
                //    adjustments = (trainingInputs^T) • (error * sigmoid'(output))
                //
                // - Activate(output, true) gives sigmoid'(output) element-wise.
                // - error * sigmoid'(output) is element-wise multiply.
                // - trainingInputs^T converts [numSamples x 3] into [3 x numSamples]
                // - DotProduct then yields [3 x 1], matching weights shape.
                //
                // This is the core backprop step for a single layer.
                double[,] adjustments =
                    DotProduct(
                        Transpose(trainingInputs),
                        PerformOperation(error, Activate(output, true), OPERATION.Multiply)
                    );

                // 4) Update weights:
                // weights = weights + adjustments
                //
                // (Many implementations include a learning rate like weights += lr * adjustments.
                // This demo code uses an implicit learning rate of 1.)
                weights = PerformOperation(weights, adjustments, OPERATION.Add);
            }
        }

        private double[,] DotProduct(double[,] matrix1, double[,] matrix2)
        {
            // Standard matrix multiplication.
            //
            // If matrix1 is [A x B] and matrix2 is [B x C],
            // result is [A x C] where:
            //   result[i,j] = sum_k matrix1[i,k] * matrix2[k,j]
            int numberOfRowsInMatrix1 = matrix1.GetLength(0);
            int numberOfColumnsInMatrix1 = matrix1.GetLength(1);

            int numberOfRowsInMatrix2 = matrix2.GetLength(0);
            int numberOfColumnsInMatrix2 = matrix2.GetLength(1);

            double[,] result = new double[numberOfRowsInMatrix1, numberOfColumnsInMatrix2];

            for (int rowInMatrix1 = 0; rowInMatrix1 < numberOfRowsInMatrix1; rowInMatrix1++)
            {
                for (int columnInMatrix2 = 0; columnInMatrix2 < numberOfColumnsInMatrix2; columnInMatrix2++)
                {
                    double sum = 0;

                    for (int columnInMatrix1 = 0; columnInMatrix1 < numberOfColumnsInMatrix1; columnInMatrix1++)
                    {
                        // Multiply a row element from matrix1 by a column element from matrix2.
                        sum += matrix1[rowInMatrix1, columnInMatrix1] * matrix2[columnInMatrix1, columnInMatrix2];
                    }

                    result[rowInMatrix1, columnInMatrix2] = sum;
                }
            }

            return result;
        }

        public double[,] Think(double[,] inputs)
        {
            // "Inference" / forward pass:
            // 1) Weighted sum: inputs • weights
            // 2) Squash through sigmoid so outputs are between 0 and 1
            return Activate(DotProduct(inputs, weights), false);
        }

        private double[,] PerformOperation(double[,] matrix1, double[,] matrix2, OPERATION operation)
        {
            // Element-wise operation between same-sized matrices.
            // Used for:
            //  - error calculation (subtract)
            //  - applying derivative to error (multiply)
            //  - weight update (add)
            int rows = matrix1.GetLength(0);
            int columns = matrix1.GetLength(1);

            double[,] result = new double[rows, columns];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    switch (operation)
                    {
                        case OPERATION.Multiply:
                            result[i, j] = matrix1[i, j] * matrix2[i, j];
                            break;

                        case OPERATION.Add:
                            result[i, j] = matrix1[i, j] + matrix2[i, j];
                            break;

                        case OPERATION.Subtract:
                            result[i, j] = matrix1[i, j] - matrix2[i, j];
                            break;
                    }
                }
            }

            return result;
        }

        private double[,] Transpose(double[,] matrix)
        {
            // Convenience transpose for a 2D array using LINQ:
            //  1) Flatten the 2D matrix to a 1D array
            //  2) Use extension method to produce a transposed 2D matrix
            //
            // Original: [rows x cols] -> Transposed: [cols x rows]
            return matrix.Cast<double>().ToArray().Transpose(matrix.GetLength(0), matrix.GetLength(1));
        }
    }

    public static class Extensions
    {
        public static double[,] Transpose(this double[] array, int rows, int columns)
        {
            // Transpose implementation for a flattened [rows x columns] matrix.
            //
            // If original index mapping is:
            //    array[row * columns + column]
            // Then transposed becomes:
            //    result[column, row]
            double[,] result = new double[columns, rows];

            for (int row = 0; row < rows; row++)
            {
                for (int column = 0; column < columns; column++)
                {
                    result[column, row] = array[row * columns + column];
                }
            }

            return result;
        }
    }
}