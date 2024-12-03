using ClosedXML.Excel;
using CsvHelper.Configuration;
using CsvHelper;
using Microsoft.Win32;
using ScottPlot.Plottables;
using ScottPlot.WPF;
using SkiaSharp;
using System.Diagnostics;
using System.IO;
using System.Net.Sockets;
using System.Runtime.CompilerServices;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Theme.WPF.Themes;
using DocumentFormat.OpenXml.Drawing.Diagrams;
using System.Data;
using System.Dynamic;
using QARBoM_Front.Components;
using System.Reflection;

namespace QARBoM_Front
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private Process? juliaProcess;

        public MainWindow()
        {
            InitializeComponent();
            StartJuliaServer();
            InitializeGeneralPlot();
            this.Closed += OnAppClosed;
        }
        private bool CancelRuntime = false;

        private List<Dictionary<string, object>>? MainGridDataRows;
        private ScottPlot.Plottables.Signal? MSEPlot;
        private ScottPlot.Plottables.Signal? AccuracyPlot;
        private string PathCSVFolder = @"C:\\Users\\lucas\\OneDrive\\Documents\\PUC\\Stone\\";

        private int GlobalMaxTextSize = 500;

        private CurveGenerator? CG;
        private void InitializeGeneralPlot()
        {
            MainPlot.Plot.Title("Learning Progression");
        }



        #region Julia setup

        private void StartJuliaServer()
        {
            // Check if the server is already running by attempting to connect
            try
            {
                using (TcpClient client = new TcpClient("127.0.0.1", 2000))
                {
                    // If we can connect, it means the server is already running



                    client.Close();

                    StartJuliaServer();
                    return;
                }
            }
            catch (SocketException)
            {

                //Lembrar de limpar arquivos temp

                JuliaStartProcess();

            }
        }

        private void JuliaStartProcess()
        {
            if (juliaProcess != null)
            {
                MessageBox.Show("Julia Process is already running", "Warning", MessageBoxButton.OK, MessageBoxImage.Exclamation);
                return;
            }

            string tempPath = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "main_tcp_server.jl");

            using (Stream? stream = Assembly.GetExecutingAssembly().GetManifestResourceStream("QARBoM_Front.Resources.main_tcp_server.jl"))
            using (FileStream fileStream = new FileStream(tempPath, FileMode.Create, FileAccess.Write))
            {
                if (stream != null) stream.CopyTo(fileStream);
            }


            // Server is not running, so we start it
            juliaProcess = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "julia",
                    Arguments = tempPath,
                    WorkingDirectory = System.IO.Path.GetDirectoryName(tempPath),
                    UseShellExecute = false,
                    CreateNoWindow = true,
                    RedirectStandardError = true,
                    RedirectStandardOutput = true

                },
                EnableRaisingEvents = true
            };

            juliaProcess.OutputDataReceived += JuliaOutputReceived;
            juliaProcess.ErrorDataReceived += JuliaOutputReceived;

            juliaProcess.Start();
            juliaProcess.BeginOutputReadLine();
            juliaProcess.BeginErrorReadLine();
        }

        private void JuliaOutputReceived(object sender, DataReceivedEventArgs e)
        {

            if (e.Data != null)
            {
                string printable = e.Data.ToString();

                if (printable.Count() > 1000)
                {

                }

                //Debug.WriteLine($"OUTPUT: {e.Data}");

                Dispatcher.Invoke(() =>
                {
                    ResultTextBlock.Text += e.Data + "\n";
                    ScrollCMD.ScrollToEnd();

                    if (CG != null) CG.UpdateListAuto(e.Data);

                    MainPlot.Refresh();
                });

            }
        }

        #endregion

        private async Task ExecuteJuliaCommand(List<string> commands)
        {
            if (juliaProcess == null) JuliaStartProcess();

            int count = 1;
            foreach (var command in commands)
            {
                if (CancelRuntime)
                {
                    CancelRuntime = false;
                    return;
                }

                try
                {
                    using (TcpClient client = new TcpClient("127.0.0.1", 2000))
                    using (NetworkStream stream = client.GetStream())
                    {
                        byte[] data = Encoding.UTF8.GetBytes(command + "\n");
                        await stream.WriteAsync(data, 0, data.Length);
                        

                        StringBuilder responseBuilder = new StringBuilder();
                        byte[] buffer = new byte[256];
                        int bytesRead = await stream.ReadAsync(buffer, 0, buffer.Length);
                        string response = Encoding.UTF8.GetString(buffer, 0, bytesRead);

                        Application.Current.Dispatcher.Invoke(() =>
                        {
                            if (ResultTextBlock.Text.Length > GlobalMaxTextSize)
                            {
                                ResultTextBlock.Text = ResultTextBlock.Text.Substring(ResultTextBlock.Text.Length - GlobalMaxTextSize);
                            }

                            ResultTextBlock.Text += $"\n>>WPF\n>>Response: {response}\n";
                        });


                        count++;
                    }
                }
                catch (Exception ex)
                {
                    MessageBox.Show("Error connecting to Julia server: " + ex.Message);
                }
            }
        }

        #region Routines

        #region Julia commands
        private async void ExecuteMainRoutine(object sender, RoutedEventArgs e)
        {
            if (!int.TryParse(TextBoxEpochs.Text, out int epochNumber))
            {
                MessageBox.Show("Invalid number of epochs");
                return;
            }

            string metrics = GenerateMetricsString();
            List<string> metricList = GenerateMetricsList(metrics);

            CG = new CurveGenerator(epochNumber, metricList, MainPlot);

            string trainType = RadioButtonClassification.IsChecked == true ? "RBMClassifier" : "RBM";
            string trainTypeTail = RadioButtonClassification.IsChecked == true ? $", {TextBoxNodes.Text}" : "";

            string cdType = ComboBoxTrainType.Text;
            string cdSteps = cdType == "CD" ? $"cd_steps = {TextBoxCDstep.Text}, " : "";



            List<string> commands = new List<string> {
                $"using QARBoM",
                //$"rbm = QARBoM.RBM({TextBoxVisibleLayers.Text},{TextBoxHiddenLayers.Text})",
                //$"QARBoM.train!(\r\n    rbm, \r\n    train_data,\r\n    CD; \r\n    n_epochs = {TextBoxEpochs.Text},  \r\n    cd_steps = 1, # number of gibbs sampling steps\r\n    learning_rate = {TextBoxLearningRate.Text}, \r\n    metrics = [MeanSquaredError], # the metrics you want to track\r\n    early_stopping = true,\r\n    file_path = \"my_cd_metrics.csv\",\r\n)"
                $"rbm = {trainType}({TextBoxVisibleLayers.Text}, {TextBoxHiddenLayers.Text}{trainTypeTail})",
                $"QARBoM.train!(" +
                $"rbm, " +
                $"vetor_de_floats_treino[1:18000], " +
                $"y_vec[1:18000], " +
                $"{cdType}; " +
                $"n_epochs = {TextBoxEpochs.Text}, " +
                cdSteps +
                $"batch_size = {TextBoxBatchSize.Text}, "+
                $"learning_rate= [{TextBoxLearningRate.Text}/(j^0.8) for j in 1:{TextBoxEpochs.Text}], " +
                $"label_learning_rate= [{TextBoxLearningRate.Text}/(j^0.8) for j in 1:{TextBoxEpochs.Text}], " +
                $"metrics = [{metrics}]);" +
                $"println(n_epochs)"
            };

            //Debug.WriteLine($"COMMANDS:\n {commands}");

            await ExecuteJuliaCommand(commands);

        }
        private async void ExecuteImportSubroutine(string pathPP, string pathAnswers)
        {




            string StonePPCommand = $"Dataset = DataFrame(CSV.File(raw\"{pathPP}\")); println(\"Data loaded. Rows: \", size(Dataset, 1), \", Columns: \", size(Dataset, 2));";
            string RespostasCommand = $"Y = DataFrame(CSV.File(raw\"{pathAnswers}\")); println(\"Data loaded. Rows: \", size(Y, 1), \", Columns: \", size(Y, 2));";

            List<string> commands = new List<string>
            {
                "using DataFrames",
                "using CSV",
                StonePPCommand,
                "vetor_de_vetores = [collect(row) for row in eachrow(Dataset)]; println(length(vetor_de_vetores))",
                RespostasCommand,
                "Y[!, \"nova_coluna\"] = ifelse.(Y[!, \"is_known_as_fraud_inferred\"] .== 0, 1, 0);  println(size(Y, 1))",
                "y_treino = Y[:, 1:2]; println(size(y_treino, 1))",
                "y_vec = [[y1, y2] for (y1, y2) in zip(y_treino[!, 1], y_treino[!, 2])]; println(length(y_vec))",
                "y_vec = Array{Vector{Float64}, 1}(y_vec);",
                "vetor_de_floats_treino = Vector{Vector{Float64}}()",
                "append!(vetor_de_floats_treino, [float.(v) for v in vetor_de_vetores]); println(length(vetor_de_floats_treino))"
            };

            await ExecuteJuliaCommand(commands);
        }
        private async void TestButton(object sender, RoutedEventArgs e)
        {

            string trainType = RadioButtonClassification.IsChecked == true ? "RBMClassifier" : "RBM";
            string trainTypeTail = RadioButtonClassification.IsChecked == true ? $", {TextBoxNodes.Text}" : "";

            string cdType = ComboBoxTrainType.Text;
            string cdSteps = cdType == "CD" ? $"cd_steps = {TextBoxCDstep.Text}, " : "";

            string metrics = GenerateMetricsString();

            List<string> commands = new List<string> {
                $"using Revise",
                $"using QARBoM",
                //$"rbm = QARBoM.RBM({TextBoxVisibleLayers.Text},{TextBoxHiddenLayers.Text})",
                //$"QARBoM.train!(\r\n    rbm, \r\n    train_data,\r\n    CD; \r\n    n_epochs = {TextBoxEpochs.Text},  \r\n    cd_steps = 1, # number of gibbs sampling steps\r\n    learning_rate = {TextBoxLearningRate.Text}, \r\n    metrics = [MeanSquaredError], # the metrics you want to track\r\n    early_stopping = true,\r\n    file_path = \"my_cd_metrics.csv\",\r\n)"
                $"rbm = {trainType}({TextBoxVisibleLayers.Text}, {TextBoxHiddenLayers.Text}{trainTypeTail})",
                $"QARBoM.train!(" +
                $"rbm, " +
                $"vetor_de_floats_treino[1:20000], " +
                $"y_vec[1:20000], " +
                $"{cdType}; " +
                $"n_epochs = {TextBoxEpochs.Text}, " +
                cdSteps +
                $"batch_size = {TextBoxBatchSize.Text}, "+
                $"learning_rate= [{TextBoxLearningRate.Text}/(j^0.8) for j in 1:{TextBoxEpochs.Text}], " +
                $"label_learning_rate= [{TextBoxLearningRate.Text}/(j^0.8) for j in 1:{TextBoxEpochs.Text}], " +
                $"metrics = [{metrics}]);" +
                $"println(n_epochs)"
            };
            await ExecuteJuliaCommand(commands);
        }

        #endregion


        #region ExcelImport

        private void ImportExcel_Click(object sender, RoutedEventArgs e)
        {
            string fileQuestions;
            string fileAnswers;
            string filePathQuestions;
            string filePathAnswers;

            // Open File Dialog to select file
            OpenFileDialog openPPDialog = new OpenFileDialog
            {
                Filter = "Excel Files (*.xlsx)|*.xlsx|CSV Files (*.csv)|*.csv",
                Title = "Select your PP file"
            };

            if (openPPDialog.ShowDialog() == true)
            {
                filePathQuestions = openPPDialog.FileName;
                fileQuestions = openPPDialog.FileName;
            }
            else return;

            OpenFileDialog openAnswersDialog = new OpenFileDialog
            {
                Filter = "Excel Files (*.xlsx)|*.xlsx|CSV Files (*.csv)|*.csv",
                Title = "Select your Answers file"
            };

            if (openAnswersDialog.ShowDialog() == true)
            {
                filePathAnswers = openAnswersDialog.FileName;

                if (filePathQuestions.EndsWith(".xlsx"))
                {
                    ImportXlsx(filePathQuestions, filePathAnswers);
                }
                else if (filePathQuestions.EndsWith(".csv"))
                {
                    ImportCsvAsync(filePathQuestions, filePathAnswers);
                }
                else
                {
                    MessageBox.Show("Unsupported file format.");
                }

                fileAnswers = openAnswersDialog.FileName;
            }
            else return;


            ExecuteImportSubroutine(fileQuestions, fileAnswers);
        }

        private void ImportXlsx(string filePath, string dataGridName)
        {
            using (var workbook = new XLWorkbook(filePath))
            {
                var worksheet = workbook.Worksheets.Worksheet(1); // Get the first worksheet
                var dataTable = new DataGrid();


            }
        }

        private void ImportCsv(string QuestionsPath, string AnswersPath)
        {
            var config = new CsvConfiguration(System.Globalization.CultureInfo.InvariantCulture)
            {
                HasHeaderRecord = true,
            };

            List<ExpandoObject> dataRows = new List<ExpandoObject>();
            List<string> fixedHeaders = new();
            List<string> headers = new();

            using (var reader = new StreamReader(QuestionsPath))
            using (var reader2 = new StreamReader(AnswersPath))
            using (var csv = new CsvReader(reader, config))
            using (var csv2 = new CsvReader(reader2, config))
            {
                csv.Read();
                csv2.Read();
                csv.ReadHeader();
                csv2.ReadHeader();



                if (csv.HeaderRecord != null) headers = csv.HeaderRecord.ToList();
                if (csv2.HeaderRecord != null) headers?.AddRange(csv2.HeaderRecord.ToList());

                if (headers == null) return;

                foreach (var header in headers)
                {
                    string headerFixed = header.Replace("ã", "a").Replace("ç", "c").Replace("õ", "o").Replace(" ", "_");

                    if (char.IsDigit(header[0]))
                    {
                        headerFixed = "_" + headerFixed;
                    }
                    if (headerFixed.Contains("is_known")) headerFixed = "help";

                    fixedHeaders.Add(headerFixed);
                }

                // Populate data rows
                while (csv.Read() && csv2.Read())
                {
                    dynamic row = new ExpandoObject();
                    IDictionary<string, object>? rowDict = row as IDictionary<string, object>;
                    if (rowDict == null) return;

                    for (int i = 0; i < headers.Count && i < fixedHeaders.Count; i++)
                    {

                        string? currentValue;

                        if (!csv.TryGetField(headers[i], out currentValue))
                        {
                            currentValue = csv2.GetField(headers[i]);
                        }

                        if (!string.IsNullOrEmpty(currentValue))
                        {
                            rowDict[fixedHeaders[i]] = currentValue;
                        }
                    }
                    dataRows.Add(row);
                }
                //if (dataGridName == "Answers") DataGridAnswers.ItemsSource = records;
                //if (dataGridName == "Questions") DataGridQuestions.ItemsSource = records;

            }


            Application.Current.Dispatcher.Invoke(() =>
            {

                DataGridQuestions.Columns.Clear();
                for (int i = 0; i < headers.Count && i < fixedHeaders.Count; i++)
                {

                    DataGridTextColumn column = new DataGridTextColumn
                    {
                        Header = headers[i],
                        Binding = new Binding(fixedHeaders[i]) // Binding directly to the property name
                    };
                    DataGridQuestions.Columns.Add(column);
                }


                DataGridQuestions.ItemsSource = dataRows;



            });


        }

        private async void ImportCsvAsync(string Questions, string Answers)
        {
            await Task.Run(() =>
            {
                ImportCsv(Questions, Answers);
            });
        }


        #endregion

        #endregion

        #region Auxiliar

        private string GenerateMetricsString()
        {
            StringBuilder sb = new StringBuilder();

            if (RadioButtonGenerative.IsChecked == true)
            {
                if (CBMse.IsChecked == true) sb.Append("MeanSquaredError");
                return sb.ToString();
            }
            else
            {
                if (CBAccuracy.IsChecked == true) sb.Append("Accuracy, ");
                if (CBPrecision.IsChecked == true) sb.Append("Precision, ");
                if (CBRecall.IsChecked == true) sb.Append("Recall, ");
                if (CBFalseP.IsChecked == true) sb.Append("FalsePositive, ");
                if (CBFalseN.IsChecked == true) sb.Append("FalseNegative, ");
                if (CBTrueP.IsChecked == true) sb.Append("TruePositive, ");
                if (CBTrueN.IsChecked == true) sb.Append("TrueNegative, ");

                if (sb.Length > 0) sb.Remove(sb.Length - 2, 2);
                return sb.ToString();
            }
        }

        private List<string> GenerateMetricsList(string metrics)
        {
            List<string> list = new();

            list = metrics.Replace(" ", "").Split(",").ToList();
            foreach (string metric in list)
            {
                metric.TrimStart();
            }

            return list;
        }

        private void UpdateMainPlot(CurveGenerator generator, List<string> metrics)
        {
            if (MainPlot.Plot.PlottableList.Count == 0)
            {
                List<Signal> plots = generator.GetPlotList();

                for (int i = 0; i < plots.Count; i++)
                {

                }
            }

            foreach (string metric in metrics)
            {
                if (metric == "Accuracy")
                {

                }
            }
        }



        #endregion

        #region Events
        private void ChangeTheme(object sender, RoutedEventArgs e)
        {
            switch (((MenuItem)sender).Uid)
            {
                case "0":
                    ThemesController.SetTheme(ThemeType.DeepDark);
                    break;
                case "1":
                    ThemesController.SetTheme(ThemeType.SoftDark);
                    break;
                case "2":
                    ThemesController.SetTheme(ThemeType.DarkGreyTheme);
                    break;
                case "3":
                    ThemesController.SetTheme(ThemeType.GreyTheme);
                    break;
                case "4":
                    ThemesController.SetTheme(ThemeType.LightTheme);
                    break;
                case "5":
                    ThemesController.SetTheme(ThemeType.RedBlackTheme);
                    break;
            }

            e.Handled = true;
        }

        private void RadioButtonGenerative_Click(object sender, RoutedEventArgs e)
        {
            GroupBoxGenerative.Visibility = Visibility.Visible;
            GroupBoxClassification.Visibility = Visibility.Collapsed;

            LabelNodes.IsEnabled = false;
            TextBoxNodes.IsEnabled = false;
        }

        private void RadioButtonClassification_Click(object sender, RoutedEventArgs e)
        {
            GroupBoxGenerative.Visibility = Visibility.Collapsed;
            GroupBoxClassification.Visibility = Visibility.Visible;

            LabelNodes.IsEnabled = true;
            TextBoxNodes.IsEnabled = true;
        }

        private void ComboBoxTrainType_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (ComboBoxTrainType.SelectedIndex == 0)
            {
                LabelCDstep.IsEnabled = true;
                TextBoxCDstep.IsEnabled = true;
            }
            else
            {
                LabelCDstep.IsEnabled = false;
                TextBoxCDstep.IsEnabled = false;
                TextBoxCDstep.Text = "";
            }
        }

        private void OnAppClosed(object? sender, EventArgs e)
        {
            // Ensure the Julia process is closed when the app exits
            if (juliaProcess != null && !juliaProcess.HasExited)
            {
                juliaProcess.Kill();
                juliaProcess.Dispose();
            }
        }
        #endregion

        private void CancelJuliaButton_Click(object sender, RoutedEventArgs e)
        {
            if (juliaProcess == null || juliaProcess.HasExited)
            {
                MessageBox.Show("No Julia process is running.", "Warning");
                return;
            }

            try
            {
                juliaProcess.Kill();
                juliaProcess.WaitForExit();
                ResultTextBlock.Text = "Julia process terminated.";
            }
            catch (Exception ex)
            {
                ResultTextBlock.Text = "Failed to terminate Julia process: " + ex.Message;
            }
        }
    }
}