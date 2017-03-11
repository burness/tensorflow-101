
// DISCLAIMER: yes this is *very* badly written. but it gets the job done! Make sure to check server.py to understand endpoints.

var linecanvas = document.getElementById("line");
var linectx = linecanvas.getContext("2d");
linectx.lineCap = "round";
linectx.lineJoin = "round";
linectx.lineWidth = 3;

var colorcanvas = document.getElementById("color");
var colorctx = colorcanvas.getContext("2d");
colorctx.lineCap = "round";
colorctx.lineJoin = "round";
colorctx.lineWidth = 15;


colorctx.beginPath();
colorctx.rect(0, 0, 512, 512);
colorctx.fillStyle = "white";
colorctx.fill();

var lastX;
var lastY;

var mouseX;
var mouseY;
var canvasOffset = $("#color").offset();
var offsetX = canvasOffset.left;
var offsetY = canvasOffset.top;
var isMouseDown = false;


function handleMouseDown(e) {
    canvasOffset = $("#color").offset();
    offsetX = canvasOffset.left;
    offsetY = canvasOffset.top;

    mouseX = parseInt(e.pageX - offsetX);
    mouseY = parseInt(e.pageY - offsetY);

    // Put your mousedown stuff here
    lastX = mouseX;
    lastY = mouseY;
    isMouseDown = true;
}

function handleMouseUp(e) {
    mouseX = parseInt(e.pageX - offsetX);
    mouseY = parseInt(e.pageY - offsetY);

    // Put your mouseup stuff here
    isMouseDown = false;
}
function handleMouseOut(e) {
    mouseX = parseInt(e.pageX - offsetX);
    mouseY = parseInt(e.pageY - offsetY);

    // Put your mouseOut stuff here
    isMouseDown = false;
}

function handleMouseMove(e)
{
    canvasOffset = $("#color").offset();
    offsetX = canvasOffset.left;
    offsetY = canvasOffset.top;

    // var x = e.pageX - offsetX;
    // var y = e.pageY - offsetY;

    mouseX = parseInt(e.pageX - offsetX);
    mouseY = parseInt(e.pageY - offsetY);
    // Put your mousemove stuff here
    if(isMouseDown)
    {
        if(mode == "pen")
        {
            linectx.beginPath();
            linectx.globalCompositeOperation = "source-over";
            linectx.moveTo(lastX, lastY);
            linectx.lineTo(mouseX, mouseY);
            linectx.stroke();
        }
        else if(mode == "eraser")
        {
            linectx.beginPath();
            linectx.globalCompositeOperation = "destination-out";
            linectx.arc(lastX, lastY, 10, 0, Math.PI * 2, false);
            linectx.fill();
        }
        else
        {
            colorctx.beginPath();
            colorctx.strokeStyle = mode;
            colorctx.globalCompositeOperation = "source-over";
            colorctx.moveTo(lastX, lastY);
            colorctx.lineTo(mouseX, mouseY);
            colorctx.stroke();
        }
        lastX = mouseX;
        lastY = mouseY;
    }
}

$("#line").mousedown(function (e) {
    handleMouseDown(e);
});
$("#line").mousemove(function (e) {
    handleMouseMove(e);
});
$("#line").mouseup(function (e) {
    handleMouseUp(e);
});
$("#line").mouseout(function (e) {
    handleMouseOut(e);
});

var mode = "pen";
$("#pen").click(function () {
    mode = "pen";
});
$("#eraser").click(function () {
    mode = "eraser";
});

$(document).keypress(function(e) {
    console.log(e.which)
    if(e.which == 100) {
        mode = "pen";
    }
    if(e.which == 101)
    {
        mode = "eraser";
    }
});

$("#uploadform").bind('submit', function (e) {
    e.preventDefault();

    console.log("Uploadin");
    var files = document.getElementById('fileselect').files;
    var formData = new FormData();
        // Loop through each of the selected files.
    for (var i = 0; i < files.length; i++)
    {
        var file = files[i];
        formData.append('img', file, file.name);
    }
    $.ajax({
        url: '/upload_toline',
        data: formData,
        processData: false,
        contentType: false,
        type: 'POST',
        success: function(result){
            var image = new Image();
            image.onload = function() {
                colorctx.beginPath();
                colorctx.rect(0, 0, 512, 512);
                colorctx.fillStyle = "white";
                colorctx.fill();
                linectx.clearRect(0, 0, 512, 512);
                linectx.drawImage(image, 0, 0);
            };
            image.src = 'data:image/png;base64,' + result;
        }
    });

    return false;
});

$("#submit").click(function () {

    // change non-opaque pixels to white
    var imgData = linectx.getImageData(0,0,512,512);
    var data = imgData.data;
    var databackup = data.slice(0);
    for(var i = 0; i < data.length; i+=4)
    {
        if(data[i+3]<255)
        {
            data[i]=255;
            data[i+1]=255;
            data[i+2]=255;
            data[i+3]=255;
        }
    }

    linectx.putImageData(imgData,0,0);

    var dataURL = linecanvas.toDataURL("image/jpg");
    var dataURLc = colorcanvas.toDataURL("image/jpg");

    imgData = linectx.getImageData(0,0,512,512);
    data = imgData.data;
    for(var i = 0; i < data.length; i++)
    {
        data[i] = databackup[i];
    }
    linectx.putImageData(imgData,0,0);
    // console.log(dataURL)

    $.ajax({
        url: '/upload_canvas',
        type: "POST",
        data: {colors: dataURLc, lines: dataURL},
        // data: {lines: "meme"},
        success: function (result) {
            // console.log("Upload complete!!");
            // console.log(result.length);
            // console.log(result);
            $('#result').html('<img src="data:image/png;base64,' + result + '" />');
        },
        error: function (error) {
            console.log("Something went wrong!");
        }
    });
});












$('#fileselect').change(function() {
  $('#uploadform').submit();
});

$(document).ready(function(){
    $("#sanaebutton").click(function(){
        $.ajax({
            url: '/standard_sanae',
            data: "nothing",
            processData: false,
            contentType: false,
            type: 'POST',
            success: function(result){
                var image = new Image();
                image.onload = function() {
                    colorctx.beginPath();
                    colorctx.rect(0, 0, 512, 512);
                    colorctx.fillStyle = "white";
                    colorctx.fill();
                    linectx.clearRect(0, 0, 512, 512);
                    linectx.drawImage(image, 0, 0);
                };
                image.src = 'data:image/png;base64,' + result;
            }
        });
    });
    $("#picassobutton").click(function(){
        $.ajax({
            url: '/standard_picasso',
            data: "nothing",
            processData: false,
            contentType: false,
            type: 'POST',
            success: function(result){
                var image = new Image();
                image.onload = function() {
                    colorctx.beginPath();
                    colorctx.rect(0, 0, 512, 512);
                    colorctx.fillStyle = "white";
                    colorctx.fill();
                    linectx.clearRect(0, 0, 512, 512);
                    linectx.drawImage(image, 0, 0);
                };
                image.src = 'data:image/png;base64,' + result;
            }
        });
    });
    $("#armsbutton").click(function(){
        console.log("clicked");
        $.ajax({
            url: '/standard_armscross',
            data: "nothing",
            processData: false,
            contentType: false,
            type: 'POST',
            success: function(result){
                var image = new Image();
                image.onload = function() {
                    colorctx.beginPath();
                    colorctx.rect(0, 0, 512, 512);
                    colorctx.fillStyle = "white";
                    colorctx.fill();
                    linectx.clearRect(0, 0, 512, 512);
                    linectx.drawImage(image, 0, 0);
                };
                image.src = 'data:image/png;base64,' + result;
            }
        });
    });
});


$(function() {
    $('#cp7').colorpicker({
        color: '#ffaa00',
        container: true,
        inline: true
    });
    $('#cp7').colorpicker().on('changeColor', function(e) {
        mode = e.color.toHex();
    });


});
