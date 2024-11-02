import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PatchscopesComponent } from './patchscopes.component';

describe('PatchscopesComponent', () => {
  let component: PatchscopesComponent;
  let fixture: ComponentFixture<PatchscopesComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [PatchscopesComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(PatchscopesComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
