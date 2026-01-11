using TorchSharp;

namespace MazeExample
{
    internal class Program
    {
        /*
         * High-level flow:
         * 1) Define a maze grid (0=wall, 1=floor, 2=goal).
         * 2) Convert the maze into a rewards table (rewards[row,col]).
         * 3) Create a Q-table (qValues[row,col,action]) initialized to 0.
         * 4) Train using Q-learning to learn the best action for each cell.
         * 5) Navigate from a start cell by repeatedly choosing the best action (greedy).
         */

        #region Properties/Attributes
        // Maze legend:
        // 0 = wall (impassable)
        // 1 = floor (walkable)
        // 2 = goal  (terminal target)
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

        // We represent actions as strings for readability. Internally we index them:
        // 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT (based on order in "actions" array).
        private const string UP = "up";
        private const string DOWN = "down";
        private const string LEFT = "left";
        private const string RIGHT = "right";

        private static readonly string[] actions = new[] { UP, DOWN, LEFT, RIGHT };

        // Reward values:
        // - Large negative for walls (punish strongly)
        // - Small negative for floor (encourage shorter paths)
        // - Large positive for goal (encourage reaching it)
        private const int WALL_REWARD_VALUE = -500;
        private const int FLOOR_REWARD_VALUE = -10;
        private const int GOAL_REWARD_VALUE = 500;

        // rewards[row,col] holds the reward value for being in that cell.
        private static int[,] rewards;

        // qValues[row, col, action] holds the learned "quality" of taking an action from that cell.
        // This is the Q-table used by Q-learning.
        private static torch.Tensor qValues;

        // Hyperparameters (tuning knobs)
        private const float EPSILON = 0.95f;          // higher -> more exploitation (in your current epsilon logic)
        private const float DISCOUNT_FACTOR = 0.8f;   // future reward importance (gamma)
        private const float LEARNING_RATE = 0.9f;     // how fast we update Q-values (alpha)
        private const int EPISODES = 1500;            // number of training episodes

        private const int START_ROW = 11;
        private const int START_COLUMN = 5;

        // Use one Random for the whole program; re-creating Random often can repeat sequences.
        private static readonly Random rng = new Random(); 
        #endregion

        public static void Main(string[] args)
        {
            // 1) Convert maze layout into rewards table
            setupRewards(maze1, WALL_REWARD_VALUE, FLOOR_REWARD_VALUE, GOAL_REWARD_VALUE);

            // 2) Create a Q-table initialized to zero values
            setupQValues(maze1);

            // 3) Train Q-values using Q-learning updates
            trainTheModel(maze1, FLOOR_REWARD_VALUE, EPSILON, DISCOUNT_FACTOR, LEARNING_RATE, EPISODES);

            // 4) After training, follow the best-learned policy from the start to print the path
            navigateMaze(maze1, START_ROW, START_COLUMN, FLOOR_REWARD_VALUE, WALL_REWARD_VALUE);

            Console.WriteLine("Done. Press ENTER to exit.");
            Console.ReadLine();
        }

        #region Behavior/Methods/Function
        /// <summary>
        /// Builds a "rewards" grid from the maze:
        /// - For each cell, store a numeric reward based on whether it is wall/floor/goal.
        /// This rewards table is what the agent uses to compute Temporal Difference updates.
        /// </summary>
        private static void setupRewards(int[,] maze, int wallValue, int floorValue, int goalValue)
        {
            int mazeRows = maze.GetLength(0);
            int mazeColumns = maze.GetLength(1);

            rewards = new int[mazeRows, mazeColumns];

            for (int row = 0; row < mazeRows; row++)
            {
                for (int col = 0; col < mazeColumns; col++)
                {
                    // Translate maze cell type into reward value
                    switch (maze[row, col])
                    {
                        case 0: // wall
                            rewards[row, col] = wallValue;
                            break;
                        case 1: // floor
                            rewards[row, col] = floorValue;
                            break;
                        case 2: // goal
                            rewards[row, col] = goalValue;
                            break;
                    }
                }
            }
        } 
        
        /// <summary>
        /// Initializes the Q-table tensor of shape [rows, cols, actions].
        /// All Q-values start at 0, meaning the agent initially has no preference.
        /// </summary>
        private static void setupQValues(int[,] maze)
        {
            int mazeRows = maze.GetLength(0);
            int mazeColumns = maze.GetLength(1);

            // Q-table: qValues[row, col, actionIndex]
            qValues = torch.zeros(mazeRows, mazeColumns, actions.Length);
        }

        /// <summary>
        /// Returns true when the current state is NOT a floor tile.
        /// In other words:
        /// - If we're on floor: training/navigation should continue.
        /// - If we're on wall or goal: episode/path ends (terminal).
        /// </summary>
        private static bool hasHitWallOrEndOfMaze(int currentRow, int currentColumn, int floorValue)
        {
            // "Continue" only while on floor tiles.
            // Wall or Goal => terminal.
            return rewards[currentRow, currentColumn] != floorValue;
        }

        /// <summary>
        /// Chooses the next action using an epsilon-greedy policy.
        ///
        /// IMPORTANT: Your current epsilon logic is:
        /// - if random < epsilon  => exploit (choose best action)
        /// - else                => explore (random action)
        ///
        /// Many tutorials define epsilon the opposite way (epsilon = explore chance),
        /// but this is totally fine as long as you're consistent.
        /// </summary>
        private static long determineNextAction(int currentRow, int currentColumn, float epsilon)
        {
            double r = rng.NextDouble();

            // Exploit: pick best action from current state based on Q-table
            if (r < epsilon)
                return torch.argmax(qValues[currentRow, currentColumn]).item<long>();

            // Explore: pick a random action
            return rng.Next(actions.Length);
        }

        /// <summary>
        /// Computes the next (row, col) after applying an action, respecting maze boundaries.
        /// This method DOES NOT check whether the destination is a wall; it only applies movement rules.
        ///
        /// (You later interpret wall/goal/floor using rewards[][] to decide when to stop.)
        /// </summary>
        private static (int nextRow, int nextColumn) moveOneSpace(int[,] maze, int currentRow, int currentColumn, long currentAction)
        {
            int mazeRows = maze.GetLength(0);
            int mazeColumns = maze.GetLength(1);

            int nextRow = currentRow;
            int nextColumn = currentColumn;

            string action = actions[currentAction];

            // Apply action if it stays within bounds
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

        /// <summary>
        /// Trains the Q-table using standard Q-learning:
        ///
        /// Q(s,a) = Q(s,a) + alpha * [ reward + gamma * max_a' Q(s',a') - Q(s,a) ]
        ///
        /// Where:
        /// - alpha = learningRate
        /// - gamma = discountFactor
        /// - s' is the next state after taking action a in state s
        ///
        /// Each episode starts at the start location and ends when reaching wall/goal (terminal).
        /// </summary>
        private static void trainTheModel(int[,] maze, int floorValue, float epsilon, float discountFactor, float learningRate, int episodes)
        {
            for (int episode = 0; episode < episodes; episode++)
            {
                Console.WriteLine("-----Starting episode " + episode + "-----");

                // Start each episode at the fixed start location
                int currentRow = START_ROW;
                int currentColumn = START_COLUMN;

                // Keep stepping until we hit a terminal tile (wall/goal)
                while (!hasHitWallOrEndOfMaze(currentRow, currentColumn, floorValue))
                {
                    // 1) Choose action (epsilon-greedy)
                    long currentAction = determineNextAction(currentRow, currentColumn, epsilon);

                    // 2) Remember previous state
                    int previousRow = currentRow;
                    int previousColumn = currentColumn;

                    // 3) Apply action to get next state
                    var nextMove = moveOneSpace(maze, currentRow, currentColumn, currentAction);
                    currentRow = nextMove.nextRow;
                    currentColumn = nextMove.nextColumn;

                    // 4) Observe reward at the new state
                    float reward = rewards[currentRow, currentColumn];

                    // 5) Q-learning update
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

        /// <summary>
        /// Uses the trained Q-table to "play" the maze from the start position.
        /// We choose actions greedily (epsilon=1.0 using your epsilon logic),
        /// record each visited coordinate, and print the resulting path.
        ///
        /// Returns: List of [row,col] pairs representing the path taken.
        /// </summary>
        private static List<int[]> navigateMaze(int[,] maze, int startRow, int startColumn, int floorValue, int wallValue)
        {
            var path = new List<int[]>();

            // If the start is not a floor tile, there's nothing to navigate.
            if (hasHitWallOrEndOfMaze(startRow, startColumn, floorValue))
                return path;

            int currentRow = startRow;
            int currentColumn = startColumn;

            // Add starting position
            path.Add(new[] { currentRow, currentColumn });

            // Continue until we hit terminal tile (wall/goal)
            while (!hasHitWallOrEndOfMaze(currentRow, currentColumn, floorValue))
            {
                // Greedy action selection (exploit)
                int nextAction = (int)determineNextAction(currentRow, currentColumn, 1.0f);

                // Move to next state
                var nextMove = moveOneSpace(maze, currentRow, currentColumn, nextAction);
                currentRow = nextMove.nextRow;
                currentColumn = nextMove.nextColumn;

                // Only record non-wall positions in the path list
                if (rewards[currentRow, currentColumn] != wallValue)
                {
                    path.Add(new[] { currentRow, currentColumn });
                }
                // else: if it's a wall, we don't add it and the loop will end next iteration
                // because hasHitWallOrEndOfMaze will return true (terminal).
            }

            // Print the moves nicely
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
        #endregion
    }
}
