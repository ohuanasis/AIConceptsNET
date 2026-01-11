using TorchSharp;

namespace MazeExample
{
    internal class Program
    {
        int[,] maze1 = {
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
            { 0 , 0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 0 , 0, 0 }  //row 11 (start position is (11, 5))
        };

        const string UP = "up";
        const string DOWN = "down";
        const string LEFT = "left";
        const string RIGHT = "right";

        string[] actions = [UP, DOWN, LEFT, RIGHT];

        int[,] rewards;

        const int WALL_REWARD_VALUE = -500;
        const int FLOOR_REWARD_VALUE = -10;
        const int GOAL_REWARD_VALUE = 500;

        void setupRewards(int[,] maze, int wallValue, int floorValue, int goalValue)
        {
            int mazeRows = maze.GetLength(0);
            int mazeCols = maze.GetLength(1);

            rewards = new int[mazeRows, mazeCols];

            for (int row = 0; row < mazeRows; row++)
            {
                for (int col = 0; col < mazeCols; col++)
                {
                    switch (maze[row, col])
                    {
                        case 0:
                            rewards[row, col] = wallValue;
                            break;
                        case 1:
                            rewards[row, col] = floorValue;
                            break;
                        case 2:
                            rewards[row, col] = goalValue;
                            break; 
                    }

                }
            }
        }

        torch.Tensor qValues;

        static void Main(string[] args)
        {


        }

    }
}
