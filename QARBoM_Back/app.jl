using Dash

app = dash()

app.layout = html_div() do
    html_h1("Hello Dash"),
    html_div("Dash: A web application framework for your data."),
    dcc_graph(
        id = "example-graph-1",
        figure = (
            data = [
                (x = ["giraffes", "orangutans", "monkeys"], y = [20, 14, 23], type = "bar", name = "SF"),
                (x = ["giraffes", "orangutans", "monkeys"], y = [12, 18, 29], type = "bar", name = "Montreal"),
            ],
            layout = (title = "Dash Data Visualization", barmode="group")
        )
    )
end

run_server(app, "139.82.146.156", debug=true)