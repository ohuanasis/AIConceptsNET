namespace ReinforcementLearning
{
    internal class Program
    {
        static void Main(string[] args)
        {
            // One RNG for the whole program (don’t new Random() in tight loops).
            var rng = new Random();

            var agent = new QLearningAgent(
                learningRate: 0.10,
                discountFactor: 0.90,
                explorationRate: 0.20, // start a bit higher, we’ll decay it
                rng: rng
            );

            int trainEpisodes = 50_000;

            for (int episode = 1; episode <= trainEpisodes; episode++)
            {
                var env = new TicTacToeEnv();

                // Simple exploration decay (optional but helpful)
                agent.ExplorationRate = Math.Max(0.05, agent.ExplorationRate * 0.9999);

                // Train one full game
                while (true)
                {
                    // ----- Agent (X) turn -----
                    var boardBeforeX = env.GetBoardCopy();
                    int actionX = agent.ChooseAction(boardBeforeX); // chooses ONLY legal actions

                    env.ApplyAction(actionX); // guaranteed legal
                    var boardAfterX = env.GetBoardCopy();

                    if (env.CheckWin('X'))
                    {
                        agent.UpdateQTable(boardBeforeX, actionX, reward: +1.0, boardAfterX, isTerminal: true);
                        break;
                    }

                    if (env.CheckDraw())
                    {
                        agent.UpdateQTable(boardBeforeX, actionX, reward: 0.0, boardAfterX, isTerminal: true);
                        break;
                    }

                    // ----- Opponent (O) turn (random) -----
                    int actionO = env.GetRandomLegalAction(rng);
                    env.ApplyAction(actionO);

                    var boardAfterO = env.GetBoardCopy();

                    if (env.CheckWin('O'))
                    {
                        // X’s last move led to a state where O can win immediately -> punish X
                        agent.UpdateQTable(boardBeforeX, actionX, reward: -1.0, boardAfterO, isTerminal: true);
                        break;
                    }

                    if (env.CheckDraw())
                    {
                        agent.UpdateQTable(boardBeforeX, actionX, reward: 0.0, boardAfterO, isTerminal: true);
                        break;
                    }

                    // Non-terminal transition reward (0). Could use small negative step cost if desired.
                    agent.UpdateQTable(boardBeforeX, actionX, reward: 0.0, boardAfterO, isTerminal: false);
                }

                // Lightweight progress reporting
                if (episode % 5000 == 0)
                {
                    Console.WriteLine($"Trained {episode:n0}/{trainEpisodes:n0} episodes | Exploration={agent.ExplorationRate:0.000}");
                }
            }

            // -------- Evaluation (no exploration) --------
            Console.WriteLine();
            Console.WriteLine("Evaluation vs random opponent (greedy policy)...");
            double oldEps = agent.ExplorationRate;
            agent.ExplorationRate = 0.0;

            int evalGames = 500;
            int wins = 0, losses = 0, draws = 0;

            for (int i = 0; i < evalGames; i++)
            {
                var env = new TicTacToeEnv();

                while (true)
                {
                    // X
                    var b = env.GetBoardCopy();
                    int aX = agent.ChooseAction(b);
                    env.ApplyAction(aX);

                    if (env.CheckWin('X')) { wins++; break; }
                    if (env.CheckDraw()) { draws++; break; }

                    // O random
                    int aO = env.GetRandomLegalAction(rng);
                    env.ApplyAction(aO);

                    if (env.CheckWin('O')) { losses++; break; }
                    if (env.CheckDraw()) { draws++; break; }
                }
            }

            Console.WriteLine($"Games: {evalGames}");
            Console.WriteLine($"Wins : {wins} ({wins * 100.0 / evalGames:0.0}%)");
            Console.WriteLine($"Loss : {losses} ({losses * 100.0 / evalGames:0.0}%)");
            Console.WriteLine($"Draw : {draws} ({draws * 100.0 / evalGames:0.0}%)");

            agent.ExplorationRate = oldEps;

            // -------- Interactive demo --------
            Console.WriteLine();
            Console.WriteLine("Play a demo game (X=agent greedy, O=random). Press Enter to step; Ctrl+C to quit.");
            agent.ExplorationRate = 0.0;

            var demo = new TicTacToeEnv();
            while (true)
            {
                demo.DisplayBoard();

                if (demo.CheckWin('X')) { Console.WriteLine("X (agent) wins."); break; }
                if (demo.CheckWin('O')) { Console.WriteLine("O (random) wins."); break; }
                if (demo.CheckDraw()) { Console.WriteLine("Draw."); break; }

                Console.ReadLine();

                // Agent X
                int ax = agent.ChooseAction(demo.GetBoardCopy());
                demo.ApplyAction(ax);

                demo.DisplayBoard();
                if (demo.CheckWin('X')) { Console.WriteLine("X (agent) wins."); break; }
                if (demo.CheckDraw()) { Console.WriteLine("Draw."); break; }

                // Opponent O
                int ao = demo.GetRandomLegalAction(rng);
                demo.ApplyAction(ao);

                if (demo.CheckWin('O')) { demo.DisplayBoard(); Console.WriteLine("O (random) wins."); break; }
                if (demo.CheckDraw()) { demo.DisplayBoard(); Console.WriteLine("Draw."); break; }
            }
        }
    }

    public sealed class TicTacToeEnv
    {
        private readonly char[,] board = new char[3, 3];
        private const char Empty = '-';

        public TicTacToeEnv()
        {
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    board[r, c] = Empty;
        }

        public void DisplayBoard()
        {
            for (int r = 0; r < 3; r++)
            {
                for (int c = 0; c < 3; c++)
                {
                    Console.Write(board[r, c]);
                    Console.Write(' ');
                }
                Console.WriteLine();
            }
            Console.WriteLine();
        }

        public char[,] GetBoardCopy()
        {
            var copy = new char[3, 3];
            Array.Copy(board, copy, board.Length);
            return copy;
        }

        public List<int> GetLegalActions()
        {
            var actions = new List<int>();
            for (int a = 0; a < 9; a++)
            {
                int r = a / 3;
                int c = a % 3;
                if (board[r, c] == Empty) actions.Add(a);
            }
            return actions;
        }

        public int GetRandomLegalAction(Random rng)
        {
            var legal = GetLegalActions();
            // should never be called if terminal, but guard anyway
            if (legal.Count == 0) return 0;
            return legal[rng.Next(legal.Count)];
        }

        // Apply action for current player token (X or O) is handled outside.
        // For this simple env, we decide based on counts: X starts, then alternate.
        public void ApplyAction(int action)
        {
            int r = action / 3;
            int c = action % 3;
            if (r < 0 || r >= 3 || c < 0 || c >= 3) throw new ArgumentOutOfRangeException(nameof(action));
            if (board[r, c] != Empty) throw new InvalidOperationException("Illegal move.");

            char nextPlayer = GetCurrentPlayer();
            board[r, c] = nextPlayer;
        }

        private char GetCurrentPlayer()
        {
            int xCount = 0, oCount = 0;
            foreach (var cell in board)
            {
                if (cell == 'X') xCount++;
                else if (cell == 'O') oCount++;
            }
            // X goes first. If equal, X to move; else O to move.
            return (xCount == oCount) ? 'X' : 'O';
        }

        public bool CheckWin(char player)
        {
            // rows/cols
            for (int i = 0; i < 3; i++)
            {
                if (board[i, 0] == player && board[i, 1] == player && board[i, 2] == player) return true;
                if (board[0, i] == player && board[1, i] == player && board[2, i] == player) return true;
            }
            // diagonals
            if (board[0, 0] == player && board[1, 1] == player && board[2, 2] == player) return true;
            if (board[0, 2] == player && board[1, 1] == player && board[2, 0] == player) return true;

            return false;
        }

        public bool CheckDraw()
        {
            // draw = no empty AND no one wins
            foreach (var cell in board)
                if (cell == Empty) return false;

            return !CheckWin('X') && !CheckWin('O');
        }
    }

    public sealed class QLearningAgent
    {
        private readonly Dictionary<string, double[]> qTable = new Dictionary<string, double[]>();
        private readonly double learningRate;
        private readonly double discountFactor;
        private readonly Random rng;

        public double ExplorationRate { get; set; }

        public QLearningAgent(double learningRate, double discountFactor, double explorationRate, Random rng)
        {
            this.learningRate = learningRate;
            this.discountFactor = discountFactor;
            ExplorationRate = explorationRate;
            this.rng = rng;
        }

        public int ChooseAction(char[,] board)
        {
            string state = GetState(board);
            var legalActions = GetLegalActions(board);

            if (legalActions.Count == 0)
                return 0;

            // Ensure state exists
            if (!qTable.ContainsKey(state))
                qTable[state] = new double[9];

            // Explore
            if (rng.NextDouble() < ExplorationRate)
                return legalActions[rng.Next(legalActions.Count)];

            // Exploit (max Q among legal actions)
            var qValues = qTable[state];
            double best = double.MinValue;
            int bestAction = legalActions[0];

            foreach (int a in legalActions)
            {
                if (qValues[a] > best)
                {
                    best = qValues[a];
                    bestAction = a;
                }
            }

            return bestAction;
        }

        public void UpdateQTable(char[,] boardBefore, int action, double reward, char[,] boardAfter, bool isTerminal)
        {
            string state = GetState(boardBefore);
            string nextState = GetState(boardAfter);

            if (!qTable.ContainsKey(state))
                qTable[state] = new double[9];

            if (!qTable.ContainsKey(nextState))
                qTable[nextState] = new double[9];

            double maxNext = 0.0;
            if (!isTerminal)
            {
                // Only consider legal actions for next state
                var legalNext = GetLegalActions(boardAfter);
                if (legalNext.Count > 0)
                {
                    maxNext = legalNext.Max(a => qTable[nextState][a]);
                }
            }

            // Q(s,a) <- (1-a)Q(s,a) + a [ r + gamma * max_a' Q(s',a') ]
            qTable[state][action] =
                (1 - learningRate) * qTable[state][action] +
                learningRate * (reward + discountFactor * maxNext);
        }

        private static string GetState(char[,] board)
        {
            // Board snapshot only. (Good enough here since X always moves on its turns in our training loop)
            // If you later allow agent to play both sides, include "player to move" in this key.
            char[] chars = new char[9];
            int idx = 0;
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    chars[idx++] = board[r, c];

            return new string(chars);
        }

        private static List<int> GetLegalActions(char[,] board)
        {
            var actions = new List<int>();
            for (int a = 0; a < 9; a++)
            {
                int r = a / 3;
                int c = a % 3;
                if (board[r, c] == '-') actions.Add(a);
            }
            return actions;
        }
    }
}