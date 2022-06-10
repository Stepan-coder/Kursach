function login() {
    var login = document.getElementById("email").value;
    var password = document.getElementById("password").value;
    if (login || password) {
        document.getElementById("session").classList.add("back");
        setTimeout(async function() {
            let res_data = await eel.login(login, password)();
            document.getElementById("app").innerHTML = res_data;
        }, 500);
    }
}

last_chapter = "File"
function transition(event) {
    var target = event.target;

    if (target.id != last_chapter) {
        target.classList.add("is-active");
        document.getElementById(last_chapter).classList.remove("is-active")

        content_last = document.getElementById("content_" + last_chapter)
        content_this = document.getElementById("content_" + target.id)

        content_last.classList.toggle("block_hidden");
        setTimeout(function() {
            content_last.style.display = "none";
            content_this.style.display = "block";
            content_this.classList.toggle("block_hidden");
        }, 300)

        last_chapter = target.id;

    }
}

async function getPathToFile() {
    let res_file = await eel.pythonFunction()();
    alert('Файлы загружены успешно!');
    document.getElementById("apps-card").insertAdjacentHTML('beforeend', res_file['preview']);
    document.getElementById(res_file['file']['type']).insertAdjacentHTML('beforeend', res_file['file']['html']);
    document.getElementById("content_Viewing").insertAdjacentHTML('beforeend', res_file['get_data']);
};


last_doc = "0"
function open_data(event) {
    var target = event.target.id;

    content_last = document.getElementById("doc_" + last_doc)
    content_this = document.getElementById("doc_" + target)

    content_last.style.display = "none";
    content_this.style.display = "block";

    last_doc = target;
}