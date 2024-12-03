using QARBoM
using Sockets

const PORT = 2000

function start_server()
    server = listen(PORT)
    println("Julia server is running on port $PORT...")
    while true
        client = accept(server)
        println("Client connected.")
        handle_client(client)
    end
end

function handle_client(client)
    try
        while !eof(client)
            # Read the command from the client
            command = readline(client)
            println("Received command: $command")
            
            # Execute the command and capture the result
            try
                parsedValue = Meta.parse(command)
                println("Parsed command: $parsedValue")

                result = eval(parsedValue)
                println("Eval completed")
                result_len = length(string(result))
                println("Response size: $result_len")

                if (result_len < 500000)
                    result_str = string(result)
                    println("\n>JULIA\n>Result: $result_str\n")
                else println("\n>JULIA\nResponse is too long to send.\n")
                end

                # Log result on the server
                
                # # Send the result back to the client with a delimiter to indicate the end of response
                #write(client, result_str)
                write(client, "Eval successful")
                flush(client)  # Ensure data is sent immediately
            catch e
                "Error: $e" # Capture any error in the eval process as a result
                size = length("$e")
                trimmed_e = SubString("$e", 1, 500) * "\n. . . . .\n\n" * SubString("$e", size - 500, size)
                println("\n\n ! ! !  ERROR ! ! ! \n\n $trimmed_e  \n\n")
                println(size)
                write(client, "Eval failed")
                flush(client)
            end
            
            # Convert the result to a string (to handle complex types)

        end
    catch e
        println("Error handling client: $e")
    finally
        close(client)
        println("Client disconnected.")
    end
end

start_server()
server.kill()
