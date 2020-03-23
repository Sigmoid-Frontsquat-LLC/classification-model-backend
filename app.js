const express = require("express");
const app = express();
const Joi = require("joi");
const os = require("os");
const fs = require("fs");

const port = process.env.PORT | 9000;

app.use(express.json({ limit: "50mb" }));

// this is home...
app.post("/", (req, res) => {
    res.setHeader("content-type", "application/json");
    res.setHeader("server", os.hostname());

    const response = {
        message: "Hello, World!",
        code: 200
    };

    res.status(response.code).send(JSON.stringify(response));
});

// this should be a GET but .Net doesn't allow
// a body to be streamed in an HTTP GET request
// grrr...
app.post("/classify", (req, res) => {
    const scheme = {
        activation: Joi.string().required(),
        optimizer: Joi.string().required(),
        image: Joi.string().optional()
    };

    const validatation = Joi.validate(req.body, scheme);

    res.setHeader("content-type", "application/json");

    if (validatation.error === undefined || validatation.error === null) {
        let buffer = Buffer.from(req.body.image, "base64");

        const hash = require("crypto")
            .createHash("md5")
            .update(Date.now().toString())
            .digest("hex");

        const path = "temp/" + hash + ".jpg";

        fs.writeFileSync(path, buffer);

        // run python code here...
        const spawn = require("child_process").spawn;

        const process = spawn("python3", [
            "./classification.py",
            "-s",
            path,
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

            fs.unlinkSync(path);

            res.status(400).end(JSON.stringify(result));
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

            fs.unlinkSync(path);

            res.status(result.code).end(JSON.stringify(result));
        });

        // get the result...
        // send back the result...
    } else {
        const response = {
            error: validatation.error.message,
            code: 400
        };

        res.status(200).end(JSON.stringify(response));
    }
});

app.listen(port, async () => {
    console.log(`Hostname ${await require("public-ip").v4()} on port ${port}`);
});
