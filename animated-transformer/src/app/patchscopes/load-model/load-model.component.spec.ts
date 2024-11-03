import { ComponentFixture, TestBed } from '@angular/core/testing';

import { LoadModelComponent } from './load-model.component';

describe('LoadModelComponent', () => {
  let component: LoadModelComponent;
  let fixture: ComponentFixture<LoadModelComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [LoadModelComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(LoadModelComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
