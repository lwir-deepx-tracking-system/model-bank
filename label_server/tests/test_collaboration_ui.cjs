// Browser integration with a test server contract. Backend concurrency is tested separately.
const fs=require('fs'),assert=require('assert/strict');
const {chromium}=require(process.env.LWIR_PLAYWRIGHT_MODULE||'playwright');
const html=fs.readFileSync(process.env.LWIR_EDITOR_HTML,'utf8');
(async()=>{
  const browser=await chromium.launch({headless:true});
  try{
    const context=await browser.newContext({viewport:{width:1300,height:900}});
    const a=await context.newPage(), b=await context.newPage(), errors=[];
    a.on('pageerror',e=>errors.push(e.message));b.on('pageerror',e=>errors.push(e.message));
    const fixture=await browser.newPage();await fixture.setContent('<canvas width="640" height="480"></canvas>');
    const png=await fixture.locator('canvas').screenshot();await fixture.close();
    let version=1,mot='1,1,20,20,120,200,0.8000,0,-1,-1\n2,1,30,20,130,210,0.8000,0,-1,-1\n3,1,35,25,130,220,0.8000,0,-1,-1\n';
    let annotations=[],review={reviewed:[],removed:[]};
    const leases=new Map();
    const snapshot=()=>({frame_claims:true,parts_supported:true,editor_schema:3,version:'v'+version,frames:3,mot,review,
      parts:{schema_version:1,image_size:[640,480],annotations}});
    function claimList(owner){
      return Array.from(leases,([token,c])=>({...c,mine:token===owner,lease_seconds:90}));
    }
    const reply=(route,body,code=200)=>route.fulfill({status:code,contentType:'application/json',body:JSON.stringify(body)});
    await context.route('**/*',async route=>{
      const url=new URL(route.request().url()),path=url.pathname;
      if(path==='/edit')return route.fulfill({contentType:'text/html',body:html});
      if(path.includes('/images/'))return route.fulfill({contentType:'image/png',body:png});
      if(path==='/api/annotations')return reply(route,snapshot());
      if(path==='/api/claims')return reply(route,{claims:claimList(url.searchParams.get('owner'))});
      if(path==='/api/claim'){
        const {owner,user,first,last}=route.request().postDataJSON();
        const collision=[...leases].filter(([token,c])=>token!==owner&&first<=c.last&&last>=c.first)
          .flatMap(([,c])=>Array.from({length:Math.min(last,c.last)-Math.max(first,c.first)+1},(_,i)=>i+Math.max(first,c.first)));
        if(collision.length)return reply(route,{error:'frame range is already claimed',frames:collision},409);
        const claim={user,first,last,lease_seconds:90};leases.set(owner,claim);
        return reply(route,{ok:true,claim,claims:claimList(owner)});
      }
      if(path==='/api/release'){
        leases.delete(route.request().postDataJSON().owner);
        return reply(route,{ok:true});
      }
      if(path==='/api/save'){
        const {owner,editor_schema,frame_patch}=route.request().postDataJSON();
        assert.equal(editor_schema,3);assert.ok(leases.has(owner));
        for(const {frame,before,after} of frame_patch){
          assert.ok(frame>=leases.get(owner).first&&frame<=leases.get(owner).last);
          const frameMot=mot.split('\n').filter(line=>line.startsWith(frame+','));
          assert.deepEqual(before.mot,frameMot);
          assert.deepEqual(before.parts,annotations.filter(row=>row.frame===frame));
          mot=mot.split('\n').filter(line=>line&&!line.startsWith(frame+',')).concat(after.mot)
            .sort((x,y)=>parseInt(x)-parseInt(y)).join('\n')+'\n';
          annotations=annotations.filter(row=>row.frame!==frame).concat(after.parts);
        }
        version++;
        return reply(route,{ok:true,boxes:3,version:'v'+version,snapshot:snapshot()});
      }
      return reply(route,{error:'unknown'},404);
    });
    await Promise.all([a.goto('http://labels.local/edit?seq=demo&user=first'),
      b.goto('http://labels.local/edit?seq=demo&user=second')]);
    for(const page of [a,b])await page.waitForFunction(()=>window.LWIR_LOADED&&!$('collab-card').hidden);
    assert.equal(await a.locator('#id-scope').inputValue(),'current');
    await a.locator('#claim-current').click();
    await a.waitForFunction(()=>$('collab-state').textContent.includes('내 범위 1~1'));
    await b.locator('#claim-current').click();
    await b.waitForFunction(()=>$('st').textContent.includes('이미 맡은 프레임'));
    await b.locator('#claim-first').fill('2');await b.locator('#claim-last').fill('2');
    await b.locator('#claim-range').click();
    await b.waitForFunction(()=>cur===2&&$('collab-state').textContent.includes('내 범위 2~2'));
    // Both tabs hold unsaved work from the same starting revision.
    const edit=async (tab,point)=>tab.evaluate(x=>{
      sel={f:cur,i:0};activePart='left_ankle';
      if(!snapshot())throw Error('cannot edit claimed frame');
      writePart({state:'visible',xy:[x,200]});refresh();
    },point);
    await edit(a,80);await edit(b,90);
    await a.locator('#newid').fill('7');await a.locator('#relabel').click();
    assert.deepEqual(await a.evaluate(()=>[boxes.get(1)[0].id,boxes.get(2)[0].id]),[7,1]);
    await a.locator('#save').click();await a.waitForFunction(()=>window.LWIR_BASE==='v2'&&!dirty);
    await b.locator('#save').click();await b.waitForFunction(()=>window.LWIR_BASE==='v3'&&!dirty);
    assert.deepEqual(await b.evaluate(()=>[partRecord(1,7).parts.left_ankle.xy,partRecord(2,1).parts.left_ankle.xy]),[[80,200],[90,200]]);
    await b.evaluate(()=>{go(2);sel={f:2,i:0};refresh();});
    await b.locator('#part-state').focus();
    const pos=await b.evaluate(()=>{const r=C.getBoundingClientRect();return {x:r.left+(view.ox+70*view.s)/view.dpr,y:r.top+(view.oy+100*view.s)/view.dpr};});
    await b.mouse.click(pos.x,pos.y);
    assert.equal(await b.evaluate(()=>document.activeElement.id),'c');
    await b.evaluate(()=>C.dispatchEvent(new KeyboardEvent('keydown',{bubbles:true,code:'KeyD',key:'ㅇ',isComposing:true,shiftKey:true})));
    assert.deepEqual(await b.evaluate(()=>[boxes.get(2)[0].x,partRecord(2,1).parts.left_ankle.xy,partRecord(1,7).parts.left_ankle.xy]),[40,[100,200],[80,200]]);
    await b.keyboard.press('Control+z');
    assert.deepEqual(await b.evaluate(()=>[boxes.get(2)[0].x,partRecord(2,1).parts.left_ankle.xy]),[30,[90,200]]);
    await b.evaluate(()=>{go(1);sel={f:1,i:0};refresh();});
    const outside=await b.evaluate(()=>snapshot());
    assert.equal(outside,false);
    const c=await context.newPage();c.on('pageerror',e=>errors.push(e.message));
    await c.goto('http://labels.local/edit?seq=demo&user=third');
    await c.waitForFunction(()=>window.LWIR_LOADED&&!$('collab-card').hidden);
    await c.evaluate(()=>{go(2);sel={f:2,i:0};refresh();C.focus();
      C.dispatchEvent(new KeyboardEvent('keydown',{bubbles:true,code:'KeyD',key:'Process',isComposing:true}));});
    await c.waitForFunction(()=>$('st').textContent.includes('이미 맡은 프레임'));
    assert.equal(await c.evaluate(()=>boxes.get(2)[0].x),30);
    await c.evaluate(()=>{go(3);sel={f:3,i:0};refresh();C.focus();
      C.dispatchEvent(new KeyboardEvent('keydown',{bubbles:true,code:'KeyD',key:'Process',isComposing:true}));});
    await c.waitForFunction(()=>boxes.get(3)[0].x===36&&$('collab-state').textContent.includes('내 범위 3~3'));
    await b.locator('#claim-release').click();await b.waitForFunction(()=>!$('collab-state').textContent.includes('내 범위'));
    await c.evaluate(()=>{go(2);sel={f:2,i:0};refresh();C.focus();
      C.dispatchEvent(new KeyboardEvent('keydown',{bubbles:true,code:'KeyD',key:'Process',isComposing:true}));});
    await c.waitForFunction(()=>boxes.get(2)[0].x===31&&$('collab-state').textContent.includes('내 범위 2~2'));
    assert.ok(mot.includes('3,1,36.00,'));
    await c.locator('#save').click();await c.waitForFunction(()=>!dirty);
    // An old manual claim on frame 1 does not prevent a whole-sequence ID change elsewhere.
    // Pending edits to the previous range are saved before attempting to claim the new range.
    await a.evaluate(()=>{go(1);sel={f:1,i:0};activePart='nose';snapshot();writePart({state:'visible',xy:[95,32]});refresh();});
    await a.evaluate(()=>{go(3);sel={f:3,i:0};refresh();});
    await a.locator('#id-scope').selectOption('all');await a.locator('#newid').fill('9');await a.locator('#relabel').click();
    await a.waitForFunction(()=>$('id-feedback').textContent.includes('원본 2번'));
    assert.deepEqual(await a.evaluate(()=>boxes.get(3).map(b=>b.id)),[1]);
    assert.ok(annotations.some(row=>row.frame===1&&row.id===7&&row.parts.nose));
    await c.locator('#claim-release').click();
    await c.waitForFunction(()=>!$('collab-state').textContent.includes('내 범위'));
    await a.locator('#relabel').click();
    await a.waitForFunction(()=>boxes.get(3)[0].id===9&&$('collab-state').textContent.includes('내 범위 2~3'));
    assert.equal(await a.evaluate(()=>cur),3);
    assert.deepEqual(await a.evaluate(()=>[boxes.get(1)[0].id,boxes.get(2)[0].id,partRecord(2,9).parts.left_ankle.xy]),[7,9,[91,200]]);
    await a.locator('#save').click();await a.waitForFunction(()=>!dirty);
    assert.ok(annotations.some(row=>row.frame===2&&row.id===9));
    assert.deepEqual(errors,[]);
    console.log('PASS: current frame ID scope, overlap rejected, disjoint saves merged and read-only outside the claimed range');
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
