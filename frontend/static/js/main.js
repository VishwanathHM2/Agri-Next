document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("calendarForm");
    if (form) {
        form.addEventListener("submit", async function (e) {
            e.preventDefault();
            const formData = new FormData(form);
            const data = {};
            formData.forEach((v, k) => data[k] = v);

            const resDiv = document.getElementById("calendarResult");
            resDiv.innerHTML = "Loading...";

            try {
                const resp = await fetch("/api/calendar", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(data)
                });
                const result = await resp.json();
                if (result.schedule) {
                    let html = `<h4>Crop Calendar for ${result.crop}</h4><ul>`;
                    result.schedule.forEach(item => {
                        html += `<li>${item.date} — ${item.task}</li>`;
                    });
                    html += "</ul>";
                    resDiv.innerHTML = html;
                } else {
                    resDiv.innerHTML = "No schedule available.";
                }
            } catch (err) {
                resDiv.innerHTML = "Error fetching calendar.";
                console.error(err);
            }
        });
    }
});