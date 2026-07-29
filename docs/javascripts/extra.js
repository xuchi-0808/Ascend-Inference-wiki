/* 把 doc-authors 从 page.content 前移到正文 h1 之后（标题下、正文前）。
   mkdocs 模板无法在 page.content 内部插入，用 JS 移动 DOM。
   兼容 instant loading（DOMContentSwitch）。 */
function moveAuthors() {
  var authors = document.querySelector(".md-typeset > .doc-authors");
  var h1 = document.querySelector(".md-typeset > h1");
  if (authors && h1 && authors.previousElementSibling !== h1) {
    h1.after(authors);
  }
}
document.addEventListener("DOMContentLoaded", moveAuthors);
document.addEventListener("DOMContentSwitch", moveAuthors);
