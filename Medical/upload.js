var express = require('express');
var app = express();
var fs = require("fs");
var path = require('path');
var getPixels = require("get-pixels");

var bodyParser = require('body-parser');
var multer = require('multer');



app.use(express.static('images'));
app.use(bodyParser.urlencoded({ extended: false}));
app.use(multer({ dest: './uploadFolder'}).single('file'));

app.get('/index.htm', function (req, res) {
    res.sendFile(__dirname + "/" + "index.htm");
})

app.post('/file_upload', function (req, res) {
    console.log(req.file.originalname);
    console.log(req.file.fieldname);
    console.log(req.file.path);
    console.log(req.file.type);
    var file = __dirname + "/" + req.file.originalname;

    fs.readFile(req.file.path, function(err, data) {
        fs.writeFile(file, data, function (err) {
            if(err){
                console.log(err);
            }else{
                response = {
                    message:'File uploaded successfully', 
                    filename:req.file.originalname
                };
                app.set("view engine", "pug");
                app.set("views", path.join(__dirname, "views"));
                app.use("/static", express.static(path.join(__dirname)));
                res.redirect(req.file.originalname)
                getPixels(req.file.originalname, function(err, pixels) {
                    if(err) {
                        console.log("Bad image path")
                        return
                    }
                    console.log("got pixels", pixels.shape.slice())
                })
            }
            console.log(response);
            res.end(JSON.stringify(response));
        });
    });
})


var server = app.listen(8081, function () {
    var host = server.address().address
    var port = server.address().port

    console.log("Example app listening  at http://%s:%s", host, port)
})