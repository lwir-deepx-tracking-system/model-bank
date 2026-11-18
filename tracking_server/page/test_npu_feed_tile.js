// Exercise the actual page control code without a running board.
const assert = require('assert');
const fs = require('fs');
const vm = require('vm');
const path = require('path');
const elements = {};
function $(name) {
  if (!elements[name]) elements[name] = {
    checked: false, disabled: false, textContent: '', title: '', listeners: {},
    addEventListener(type, fn) { this.listeners[type] = fn; },
  };
  return elements[name];
}
let sent = [];
const feed = {requested:'arrival', mode:'arrival', active:true, pending:false,
  tile_available:true, tile_active:false, can_toggle:true, cores:[10,10]};
const ctx = {
  $, rlast:{npu_feed:feed, server_restart:true}, SM:{textContent:''},
  document:{body:{classList:{contains:()=>false}}},
  fetch:async (url, options) => {sent.push([url, options]);return {json:async()=>({ok:true})};},
  rtick:async()=>{}, confirm:()=>false, setTimeout, clearTimeout, console,
};
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(path.join(__dirname, 'js/13_system_buttons.js'), 'utf8'), ctx);

async function main() {
  ctx.syncSystemButtons();
  assert.equal($('npu-feed').checked, true);
  assert.equal($('npu-feed-tile').disabled, false);
  $('npu-feed-tile').checked = true;
  $('npu-feed-tile').listeners.change();
  assert.equal($('npu-feed').checked, false);
  assert.equal($('npu-feed-tile').checked, true);
  assert.equal($('npu-feed-apply').disabled, false);
  await $('npu-feed-apply').listeners.click();
  assert.equal(sent[0][0], '/system/npu-feed?mode=tile');
  Object.assign(feed, {requested:'tile', mode:'tile', active:false, tile_active:true});
  ctx.syncSystemButtons();
  assert.match($('npu-feed-state').textContent, /한 장을 나눠/);
  assert.equal($('npu-feed-apply').disabled, true);
  $('npu-feed-tile').checked = false;
  $('npu-feed-tile').listeners.change();
  await $('npu-feed-apply').listeners.click();
  assert.equal(sent[1][0], '/system/npu-feed?mode=normal');
  Object.assign(feed, {requested:'normal', mode:'normal', tile_active:false, tile_available:false});
  ctx.syncSystemButtons();
  assert.equal($('npu-feed-tile').disabled, true);
  console.log('Tile GUI selection, request, current mode and fallback: OK');
}
main().catch(error=>{console.error(error);process.exitCode=1;});
