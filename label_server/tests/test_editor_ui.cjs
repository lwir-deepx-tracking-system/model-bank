// Isolated browser QA. All HTTP is mocked; no camera, labels or device access.
// LWIR_PLAYWRIGHT_MODULE points to an installed Playwright package.
const fs=require('fs'),path=require('path'),assert=require('assert/strict');
const {chromium}=require(process.env.LWIR_PLAYWRIGHT_MODULE||'playwright');
const root=path.resolve(__dirname,'..');
const html=fs.readFileSync(process.env.LWIR_EDITOR_HTML||path.join(root,'page/label_editor.html'),'utf8');
const web=html.includes('window.LWIR_SERVER_MODE=true');
const mot=Array.from({length:3},(_,i)=>`${i+1},1,50,40,240,390,0.9000,0,-1,-1\n${i+1},2,350,40,220,390,0.8000,0,-1,-1`).join('\n')+'\n';
const empty={schema_version:1,image_size:[640,480],annotations:[]};
const img='<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480"><rect width="640" height="480" fill="#303842"/><text x="80" y="460" fill="#ddd">Synthetic annotation test</text><circle cx="160" cy="100" r="32" fill="#999"/><rect x="110" y="140" width="110" height="200" fill="#777"/></svg>';
(async()=>{
 const browser=await chromium.launch({headless:true});
 try{
  const fixture=await browser.newPage();await fixture.setContent(img);const png=await fixture.locator('svg').screenshot();await fixture.close();
  const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];let saves=[],version='v1',conflict=false,holdSave=null,releaseSave,acceptDialogs=true,dialogs=0,imagePaths=[];
  page.on('pageerror',e=>errors.push(String(e)));page.on('dialog',d=>{dialogs++;return acceptDialogs?d.accept():d.dismiss();});
  await page.route('**/*',async route=>{
   const u=new URL(route.request().url());let body;
   if(u.pathname==='/edit')return route.fulfill({contentType:'text/html',body:html});
   if(u.pathname.includes('/images/')){imagePaths.push(u.pathname);return route.fulfill({contentType:'image/png',body:png});}
   if(u.pathname==='/api/annotations')body={version,frames:3,mot,review:{},parts:empty,parts_supported:true};
   else if(u.pathname==='/api/meta')body={version,open_by:[]};
   else if(u.pathname==='/api/save'){
    const req=route.request().postDataJSON();saves.push(req);
    if(holdSave)await holdSave;
    if(conflict)return route.fulfill({status:409,contentType:'application/json',body:JSON.stringify({error:'conflict',version:'other'})});
    version='v'+(saves.length+1);body={ok:true,version,boxes:6};
   }else if(u.pathname.endsWith('/RESULT.json'))body={frames:3};
   else if(u.pathname.endsWith('/tracks_mot.txt'))return route.fulfill({body:mot});
   else return route.fulfill({status:404,body:''});
   return route.fulfill({contentType:'application/json',body:JSON.stringify(body)});
  });
  await page.goto('http://labels.local/edit?src=/data/'+(web?'wrong-sequence':'demo')+'/&seq=demo&user=test');
  await page.waitForFunction(()=>nFrames===3&&!window.LWIR_LOAD_FAILED);
  assert.ok((await page.locator('#st').textContent()).includes(web?'서버에 씁니다':'파일로 내려받습니다'));
  const initial=await page.evaluate(()=>toMot());
  const xy=async(x,y)=>page.evaluate(([x,y])=>{const r=C.getBoundingClientRect();return{x:r.left+(view.ox+x*view.s)/view.dpr,y:r.top+(view.oy+y*view.s)/view.dpr};},[x,y]);
  const clickAt=async(x,y)=>{const p=await xy(x,y);await page.mouse.click(p.x,p.y);};
  const selectPart=async key=>page.locator(`[data-part="${key}"]`).click();
  await clickAt(170,220);
  await page.locator('#part-place').click();let a=await xy(120,60),b=await xy(200,130);
  await page.mouse.move(a.x,a.y);await page.mouse.down();await page.mouse.move(b.x,b.y);await page.mouse.up();
  let doc=await page.evaluate(()=>partsDocument());assert.equal(doc.annotations.length,1);assert.equal(doc.annotations[0].parts.head.state,'visible');assert.ok(Math.abs(doc.annotations[0].parts.head.xywh[2]-80)<1);
  await selectPart('left_shoulder');await page.locator('#part-place').click();await clickAt(120,155);
  await selectPart('torso');await page.locator('#part-state').selectOption('occluded');
  doc=await page.evaluate(()=>partsDocument());assert.equal(doc.annotations[0].parts.torso.xywh,null);assert.equal(doc.annotations[0].parts.torso.state,'occluded');
  await selectPart('left_hip');await page.locator('#part-state').selectOption('out_of_frame');
  assert.equal(await page.evaluate(()=>toMot()),initial,'parts must not change MOT/person count');
  await page.locator('#newid').fill('7');await page.locator('#relabel').click();
  assert.equal((await page.evaluate(()=>partsDocument())).annotations[0].id,7);
  await page.locator('#c').click({position:{x:5,y:5}});await page.keyboard.press('Control+z');
  assert.equal((await page.evaluate(()=>partsDocument())).annotations[0].id,1);
  await clickAt(170,220);await page.locator('#next').click();await page.locator('#parts-copy').click();
  doc=await page.evaluate(()=>partsDocument());assert.equal(doc.annotations[1].frame,2);assert.equal(doc.annotations[1].needs_review,true);
  await page.locator('#parts-confirm').click();assert.equal((await page.evaluate(()=>partsDocument())).annotations[1].needs_review,false);
  acceptDialogs=false;const ignoredBefore=await page.evaluate(()=>JSON.stringify([toMot(),partsDocument()]));await page.locator('#igtoggle').click();assert.equal(await page.evaluate(()=>JSON.stringify([toMot(),partsDocument()])),ignoredBefore,'cancel ignore preserves parts');acceptDialogs=true;
  await page.locator('#igtoggle').click();assert.equal((await page.evaluate(()=>partsDocument())).annotations.length,1);
  await page.locator('#c').click({position:{x:5,y:5}});await page.keyboard.press('Control+z');assert.equal((await page.evaluate(()=>partsDocument())).annotations.length,2);
  await clickAt(170,220);await page.keyboard.press('Control+c');await page.locator('#next').click();await page.keyboard.press('Control+v');
  doc=await page.evaluate(()=>partsDocument());assert.equal(doc.annotations.length,3);assert.equal(doc.annotations[2].needs_review,true);
  await page.locator('#newid').fill('2');await page.locator('#swapid').click();assert.equal((await page.evaluate(()=>partsDocument())).annotations[2].id,2);
  await page.locator('#delbox').click();assert.equal((await page.evaluate(()=>partsDocument())).annotations.length,2);
  // Copying a box with no parts must preserve the destination's annotation.
  await page.evaluate(()=>{const before=JSON.stringify(partRecord(1,1).parts);copyPartTo(null,1,1);if(JSON.stringify(partRecord(1,1).parts)!==before||!partRecord(1,1).needs_review)throw Error('empty source lost parts');});
  // Ignored source box: cancellation must leave the current person and parts intact.
  await page.evaluate(()=>{go(1);sel={f:1,i:0};cbuf=[{...boxes.get(1)[0],g:true}];cbufParts={};cbufF=3;refresh();});
  acceptDialogs=false;const beforeCopy=await page.evaluate(()=>JSON.stringify([toMot(),partsDocument()])),dialogBefore=dialogs;
  await page.locator('#c').focus();await page.keyboard.press('Control+v');assert.equal(dialogs,dialogBefore+1);assert.equal(await page.evaluate(()=>JSON.stringify([toMot(),partsDocument()])),beforeCopy,'ignored copy must ask before deleting parts');
  acceptDialogs=true;await page.keyboard.press('Control+v');assert.equal((await page.evaluate(()=>partsDocument())).annotations.length,1);await page.keyboard.press('Control+z');assert.equal(await page.evaluate(()=>JSON.stringify([toMot(),partsDocument()])),beforeCopy);
  if(web){
   const beforeImport=await page.evaluate(()=>JSON.stringify([srcName,toMot(),partsDocument()]));
   await page.evaluate(async()=>{await loadFromFiles([]);await loadFromDir({name:'wrong-sequence'});});
   assert.equal(await page.evaluate(()=>JSON.stringify([srcName,toMot(),partsDocument()])),beforeImport,'server editor must not import another sequence');
   assert.ok(imagePaths.length&&imagePaths.every(p=>p.startsWith('/data/demo/')),'seq must determine image source');
   await page.locator('#save').click();await page.waitForFunction(()=>!dirty);assert.equal(saves[0].parts.annotations.length,2);assert.equal(saves[0].base,'v1');
   // Changes made during an in-flight save remain dirty and retain the new base.
   await page.evaluate(()=>{go(1);sel={f:1,i:0};refresh();});await selectPart('right_hip');await page.locator('#part-state').selectOption('unknown');
   holdSave=new Promise(r=>releaseSave=r);await page.locator('#save').click();await page.waitForTimeout(50);
   await selectPart('right_wrist');await page.locator('#part-state').selectOption('occluded');releaseSave();holdSave=null;
   await page.waitForFunction(()=>window.LWIR_BASE==='v3');assert.equal(await page.evaluate(()=>dirty),true);
   conflict=true;await page.locator('#save').click();await page.waitForFunction(()=>$('st').textContent.includes('덮어쓰지'));assert.equal(await page.evaluate(()=>dirty),true);assert.equal(await page.evaluate(()=>window.LWIR_BASE),'v3');
  }
  if(!web){
   const downloads=[];page.on('download',d=>downloads.push(d.suggestedFilename()));
   await page.evaluate(()=>{dirHandle={getFileHandle(){throw Error('offline export must not overwrite originals');}};});
   await page.locator('#save').click();await page.waitForTimeout(100);
   assert.deepEqual(downloads.sort(),['person_parts.json','review_state.json','tracks_edited_mot.txt']);
   assert.equal(await page.evaluate(()=>dirty),true,'offline export must retain recovery draft');
  }
  const output=process.env.LWIR_TEST_OUTPUT;
  await page.evaluate(()=>{go(1);sel={f:1,i:0};activePart='head';refresh();status('신체 부위 편집 예시 · 합성 영상 · 실제 라벨 자료 없음');});await page.locator('#parts-card').scrollIntoViewIfNeeded();
  if(output){fs.mkdirSync(output,{recursive:true});await page.screenshot({path:path.join(output,web?'parts-web-desktop.png':'parts-local-desktop.png')});}
  await page.setViewportSize({width:430,height:900});await page.waitForTimeout(50);
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'mobile overflow');
  assert.ok(await page.evaluate(()=>C.getBoundingClientRect().bottom<=document.querySelector('.bar').getBoundingClientRect().top),'canvas must not cover playback controls');
  assert.ok(await page.evaluate(()=>document.querySelector('.view').getBoundingClientRect().bottom<=document.querySelector('.side').getBoundingClientRect().top),'mobile editor must not overlap part controls');
  if(output)await page.screenshot({path:path.join(output,web?'parts-web-mobile.png':'parts-local-mobile.png'),fullPage:true});
  assert.deepEqual(errors,[]);console.log('PASS: parts geometry, person count, ID, undo, copy/review, ignore/cancel, empty/ignored source, swap/delete, '+(web?'save/race/conflict/import guard, ':'')+'desktop/mobile');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
