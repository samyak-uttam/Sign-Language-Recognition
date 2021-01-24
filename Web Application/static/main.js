var character = document.querySelector('#character');
var word = document.querySelector('#word');
var sentence = document.querySelector('#sentence');
var position = 0;
            
var xhr = new XMLHttpRequest();
xhr.open('GET', "{{ url_for('results_feed') }}");
xhr.send();
function handleNewdata() {
    var messages = (xhr.responseText.split("\n")[position]).split(" ");
    
    character.innerText = messages[0];
    word.innerText = messages[1];
    sentence.innerText = messages[2];
    console.log(messages);
    position += 1;
}

var timer;
timer = setInterval(function() {
    handleNewdata();
    if (xhr.readyState == XMLHttpRequest.DONE) {
        clearInterval(timer);
    }
}, 1000);