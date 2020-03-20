const express = require("express");
const app = express();
const Joi = require("joi");
const os = require("os");

const port = process.env.PORT | 9000;

app.use(express.json());

// this is home...
app.get("/", (req, res) => {
    res.setHeader("content-type", "application/json");
    res.setHeader("server", os.hostname());

    const response = {
        message: "Hello, World!",
        code: 200
    };

    res.status(response.code).send(JSON.stringify(response));
});

app.get("/classify", (req, res) => {
    const scheme = {
        activation: Joi.string().required(),
        optimizer: Joi.string().required()
    };

    const validatation = Joi.validate(req.body, scheme);

    res.setHeader("content-type", "application/json");

    if (validatation.error === undefined || validatation.error === null) {
        // run python code here...
        const spawn = require("child_process").spawn;

        const process = spawn("python3", [
            "./classification.py",
            "-a",
            req.body.activation,
            "-o",
            req.body.optimizer
        ]);

        let error = "";

        process.stderr.on("data", chunk => {
            error += chunk;
        });

        process.stderr.on("end", () => {
            const result = {};

            if (error === "") return;

            result.output = error;
            result.code = 400;

            res.status(result.code).end(JSON.stringify(result));
        });

        let output = "";

        process.stdout.on("data", chunk => {
            output += chunk;
        });

        process.stdout.on("end", () => {
            const result = {};
            // we have gotten the final chunk of the stream, now
            // process the output.
            result.output = output;
            result.code = 200;

            res.status(result.code).end(JSON.stringify(result));
        });

        // get the result...
        // send back the result...
    } else {
        res.status(400).end(validatation.error.message);
    }
});

app.listen(port, () => {
    console.log(`Listening on port ${port}`);
});
