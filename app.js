const express = require("express");
const app = express();

const port = process.env.PORT | 9000;

app.use(express.json());

// this is home...
app.get("/", (req, res) => {
    res.setHeader("content-type", "application/json");

    const response = {
        message: "Hello, World!",
        code: 200
    };

    res.status(response.code).send(JSON.stringify(response));
});

app.listen(port, () => {
    console.log(`Listening on port ${port}`);
});
