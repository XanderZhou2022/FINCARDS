document.addEventListener("DOMContentLoaded", function () {
  var burgers = Array.prototype.slice.call(document.querySelectorAll(".navbar-burger"), 0);

  burgers.forEach(function (burger) {
    burger.addEventListener("click", function () {
      var targetId = burger.dataset.target;
      var target = document.getElementById(targetId);

      burger.classList.toggle("is-active");
      if (target) {
        target.classList.toggle("is-active");
      }
    });
  });

  document.querySelectorAll(".navbar-menu a").forEach(function (link) {
    link.addEventListener("click", function () {
      document.querySelectorAll(".navbar-burger, .navbar-menu").forEach(function (element) {
        element.classList.remove("is-active");
      });
    });
  });

  function copyText(text) {
    function fallbackCopy() {
      var textArea = document.createElement("textarea");
      textArea.value = text;
      textArea.setAttribute("readonly", "");
      textArea.style.position = "fixed";
      textArea.style.left = "-9999px";
      document.body.appendChild(textArea);
      textArea.select();

      return new Promise(function (resolve, reject) {
        try {
          document.execCommand("copy");
          resolve();
        } catch (error) {
          reject(error);
        } finally {
          document.body.removeChild(textArea);
        }
      });
    }

    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text).catch(fallbackCopy);
    }

    return fallbackCopy();
  }

  function showCopyFeedback(button, textSpan, label) {
    button.classList.add("is-success");
    if (textSpan) {
      textSpan.textContent = label;
    }

    window.setTimeout(function () {
      button.classList.remove("is-success");
      if (textSpan) {
        textSpan.textContent = "Copy";
      }
    }, 1800);
  }

  document.querySelectorAll("[data-copy-target]").forEach(function (button) {
    button.addEventListener("click", function () {
      var target = document.getElementById(button.dataset.copyTarget);
      var textSpan = button.querySelector("span:last-child");

      if (!target) {
        return;
      }

      copyText(target.innerText).then(function () {
        showCopyFeedback(button, textSpan, "Copied");
      }).catch(function () {
        showCopyFeedback(button, textSpan, "Copy failed");
      });
    });
  });

  var scrollTopButton = document.getElementById("scroll-top-button");

  if (scrollTopButton) {
    window.addEventListener("scroll", function () {
      scrollTopButton.classList.toggle("is-visible", window.scrollY > 420);
    });

    scrollTopButton.addEventListener("click", function () {
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }
});
