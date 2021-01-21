var video = document.querySelector("#videoElement");

navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUsermedia 
        || navigator.mozGetUserMedia || navigato.msGetUserMedia || navigator.oGetUserMedia;
if (navigator.getUserMedia) {
	navigator.getUserMedia({
			video: true
		},
		handleVideo,
		videoError
	)
}

function handleVideo(stream) {
	video.srcObject = stream;
	video.play();
}

function videoError(e) {
	console.log("Something went wrong!")
}
