// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * Hides Sphinx auto-generated attribute links from the "On this page" sidebar.
 * @return {boolean} Whether attributes were found and hidden from the TOC.
 */
function hideAttributesFromTOC() {
  const attributeIds = new Set(
      Array.from(
          document.querySelectorAll('dl.py.class > dd > dl.py.attribute > dt[id]'),
          dt => dt.id));
  if (attributeIds.size === 0) return false;

  let hiddenCount = 0;
  document
      .querySelectorAll(
          '.page-toc a[href^="#"], .bd-toc-nav a[href^="#"], .toc-tree a[href^="#"]'
      )
      .forEach(link => {
    if (attributeIds.has(link.getAttribute('href').substring(1))) {
      const listItem = link.closest('li');
      if (listItem && listItem.style.display !== 'none') {
        listItem.style.display = 'none';
        hiddenCount++;
        const parentList = listItem.parentElement;
        if (parentList && parentList.tagName === 'UL' &&
            Array.from(parentList.children).every(
                el => el.style.display === 'none')) {
          parentList.style.display = 'none';
        }
      }
    }
  });
  return hiddenCount > 0;
}

if (!hideAttributesFromTOC() && document.readyState === 'loading') {
  const observer = new MutationObserver((mutations, obs) => {
    if (hideAttributesFromTOC() || document.readyState !== 'loading') {
      obs.disconnect();
    }
  });
  observer.observe(document.documentElement, {childList: true, subtree: true});
}
