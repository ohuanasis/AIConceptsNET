using TorchSharp;

namespace MazeExample
{
    internal class Program
    {
        // Maze legend:
        // 0 = wall
        // 1 = floor
        // 2 = goal
        private static readonly int[,] maze1 =
        {
            //0   1   2   3   4   5   6   7   8   9   10  11
            { 0 , 0 , 0 , 0 , 0 , 2 , 0 , 0 , 0 , 0 , 0 , 0 }, //row 0
            { 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 }, //row 1
            { 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 1 , 0 }, //row 2
            { 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 }, //row 3
            { 0 , 0 , 0 , 0 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 0 }, //row 4
            { 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 }, //row 5
            { 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 }, //row 6
            { 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 1 , 0 }, //row 7
            { 0 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 }, //row 8
            { 0 , 1 , 0 , 1 , 0 , 0 , 0 , 1 , 0 , 1 , 1 , 0 }, //row 9
            { 0 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 0 }, //row 10
            { 0 , 0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 }  //row 11 (start position is (11, 5))
        };

        private const string UP = "up";
        private const string DOWN = "down";
        private const string LEFT = "left";
        private const string RIGHT = "right";

        private static readonly string[] actions = new[] { UP, DOWN, LEFT, RIGHT };

        private const int WALL_REWARD_VALUE = -500;
        private const int FLOOR_REWARD_VALUE = -10;
        private const int GOAL_REWARD_VALUE = 500;

        private static int[,] rewards;
        private static torch.Tensor qValues;

        // Hyperparameters
        private const float EPSILON = 0.95f;
        private const float DISCOUNT_FACTOR = 0.8f;
        private const float LEARNING_RATE = 0.9f;
        private const int EPISODES = 1500;

        private const int START_ROW = 11;
        private const int START_COLUMN = 5;

        // Keep one Random for the whole program (otherwise you can get repeated sequences)
        private static readonly Random rng = new Random();

        public static void Main(string[] args)
        {
            setupRewards(maze1, WALL_REWARD_VALUE, FLOOR_REWARD_VALUE, GOAL_REWARD_VALUE);
            setupQValues(maze1);

            trainTheModel(maze1, FLOOR_REWARD_VALUE, EPSILON, DISCOUNT_FACTOR, LEARNING_RATE, EPISODES);
            navigateMaze(maze1, START_ROW, START_COLUMN, FLOOR_REWARD_VALUE, WALL_REWARD_VALUE);

            Console.WriteLine("Done. Press ENTER to exit.");
            Console.ReadLine();
        }

        private static void setupRewards(int[,] maze, int wallValue, int floorValue, int goalValue)
        {
            int mazeRows = maze.GetLength(0);
            int mazeColumns = maze.GetLength(1);

            rewards = new int[mazeRows, mazeColumns];

            for (int i = 0; i < mazeRows; i++)
            {
                for (int j = 0; j < mazeColumns; j++)
                {
                    switch (maze[i, j])
                    {
                        case 0:
                            rewards[i, j] = wallValue;
                            break;
                        case 1:
                            rewards[i, j] = floorValue;
                            break;
                        case 2:
                            rewards[i, j] = goalValue;
                            break;
                    }
                }
            }
        }

        private static void setupQValues(int[,] maze)
        {
            int mazeRows = maze.GetLength(0);
            int mazeColumns = maze.GetLength(1);

            // Q-table: [row, col, action]
            qValues = torch.zeros(mazeRows, mazeColumns, 4);
        }

        private static bool hasHitWallOrEndOfMaze(int currentRow, int currentColumn, int floorValue)
        {
            // In this setup, "continue episode" only while we are on floor cells.
            // Wall/Goal are terminal.
            return rewards[currentRow, currentColumn] != floorValue;
        }

        private static long determineNextAction(int currentRow, int currentColumn, float epsilon)
        {
            // NOTE: Your original logic is reversed vs the usual epsilon-greedy naming:
            // - if random < epsilon -> choose argmax (exploit)
            // - else -> random action (explore)
            // Kept as-is to preserve your behavior.
            double r = rng.NextDouble();

            if (r < epsilon)
                return torch.argmax(qValues[currentRow, currentColumn]).item<long>();

            return rng.Next(4);
        }

        private static (int nextRow, int nextColumn) moveOneSpace(int[,] maze, int currentRow, int currentColumn, long currentAction)
        {
            int mazeRows = maze.GetLength(0);
            int mazeColumns = maze.GetLength(1);

            int nextRow = currentRow;
            int nextColumn = currentColumn;

            string action = actions[currentAction];

            if (action == UP && currentRow > 0)
                nextRow--;
            else if (action == DOWN && currentRow < mazeRows - 1)
                nextRow++;
            else if (action == LEFT && currentColumn > 0)
                nextColumn--;
            else if (action == RIGHT && currentColumn < mazeColumns - 1)
                nextColumn++;

            return (nextRow, nextColumn);
        }

        private static void trainTheModel(int[,] maze, int floorValue, float epsilon, float discountFactor, float learningRate, int episodes)
        {
            for (int episode = 0; episode < episodes; episode++)
            {
                Console.WriteLine("-----Starting episode " + episode + "-----");

                int currentRow = START_ROW;
                int currentColumn = START_COLUMN;

                while (!hasHitWallOrEndOfMaze(currentRow, currentColumn, floorValue))
                {
                    long currentAction = determineNextAction(currentRow, currentColumn, epsilon);

                    int previousRow = currentRow;
                    int previousColumn = currentColumn;

                    var nextMove = moveOneSpace(maze, currentRow, currentColumn, currentAction);
                    currentRow = nextMove.nextRow;
                    currentColumn = nextMove.nextColumn;

                    float reward = rewards[currentRow, currentColumn];
                    float previousQValue = qValues[previousRow, previousColumn, currentAction].item<float>();

                    float maxNextQ = torch.max(qValues[currentRow, currentColumn]).item<float>();
                    float temporalDifference = reward + (discountFactor * maxNextQ) - previousQValue;

                    float nextQValue = previousQValue + (learningRate * temporalDifference);
                    qValues[previousRow, previousColumn, currentAction] = nextQValue;
                }

                Console.WriteLine("-----Finished episode " + episode + "-----");
            }

            Console.WriteLine("Completed training!");
        }

        private static List<int[]> navigateMaze(int[,] maze, int startRow, int startColumn, int floorValue, int wallValue)
        {
            var path = new List<int[]>();

            if (hasHitWallOrEndOfMaze(startRow, startColumn, floorValue))
                return path;

            int currentRow = startRow;
            int currentColumn = startColumn;

            path.Add(new[] { currentRow, currentColumn });

            while (!hasHitWallOrEndOfMaze(currentRow, currentColumn, floorValue))
            {
                // epsilon=1.0 here means "always exploit" with your determineNextAction implementation
                int nextAction = (int)determineNextAction(currentRow, currentColumn, 1.0f);

                var nextMove = moveOneSpace(maze, currentRow, currentColumn, nextAction);
                currentRow = nextMove.nextRow;
                currentColumn = nextMove.nextColumn;

                if (rewards[currentRow, currentColumn] != wallValue)
                {
                    path.Add(new[] { currentRow, currentColumn });
                }
                // else: hit wall; you "continue" (kept behavior)
            }

            int moveCount = 1;
            for (int i = 0; i < path.Count; i++)
            {
                Console.Write("Move " + moveCount + ": (");
                foreach (int element in path[i])
                {
                    Console.Write(" " + element);
                }
                Console.WriteLine(" )");
                moveCount++;
            }

            return path;
        }
    }
}